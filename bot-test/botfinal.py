import asyncio
from dataclasses import dataclass
import hashlib
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Enable UTF‑8 on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()
from playwright.async_api import async_playwright
from database import confirm_delivery, delivery_state, init_database, reserve_delivery, save_message
from rag_client import BridgeError, analyze, format_answer, health


# ============================================================
# CONFIGURAÇÃO E AMBIENTE
# ============================================================

# Modo de simulação (Dry-Run):
MODO_DRY_RUN = os.getenv("WHATSAPP_SEND_ENABLED", "false").lower() != "true"
CONTATOS_AUTORIZADOS = {name.strip() for name in os.getenv("WHATSAPP_ALLOWED_CONTACTS", "").split(",") if name.strip()}
FILA_MAXIMA = max(1, int(os.getenv("RAG_QUEUE_MAX", "10")))
PROCESSAR_NAO_LIDAS_ANTIGAS = os.getenv("BOT_PROCESS_EXISTING_UNREAD", "false").lower() == "true"
MODO_CHAT_UNICO = os.getenv("BOT_SCAN_MODE", "single_chat") == "single_chat"
CHAT_ALVO = os.getenv("BOT_TARGET_CHAT", "").strip()


def chat_corresponde_alvo(nome: str) -> bool:
    if nome == CHAT_ALVO:
        return True
    # O WhatsApp pode mudar espaços, parênteses e hífen na exibição do número.
    if re.fullmatch(r"\+?[\d\s().-]+", nome) and re.fullmatch(r"\+?[\d\s().-]+", CHAT_ALVO):
        return re.sub(r"\D", "", nome) == re.sub(r"\D", "", CHAT_ALVO)
    return False


def delivery_key(chat_name: str, message_id: str) -> str:
    return hashlib.sha256(f"{chat_name}:{message_id}".encode("utf-8")).hexdigest()

PERFIL_WHATSAPP = "whatsapp_profile"
INTERVALO_SCANNER = 1.0       # Frequência de varredura de novas mensagens (segundos)
NUM_WORKERS_IA = 1           # Ollama local: uma análise por vez
LIMITE_CONTEXTO = 15          # Mensagens recentes no histórico para dar contexto à IA
IGNORAR_GRUPOS = True         # Se True, ignora conversas e mensagens de grupos



# ============================================================
# ESTRUTURAS DE DADOS (FILAS E MENSAGENS)
# ============================================================

@dataclass
class MensagemTarefa:
    chat_nome: str
    mensagem_id: str
    texto: str
    metadata: str
    contexto: List[Dict[str, str]]


@dataclass
class RespostaTarefa:
    chat_nome: str
    mensagem_id: str
    resposta_texto: str


# ============================================================
# PARSER E EXTRAÇÃO DO DOM DO WHATSAPP (ASSÍNCRONO)
# ============================================================

async def eh_mensagem_de_audio(mensagem) -> bool:
    """Verifica se o elemento da mensagem é uma mensagem de voz ou áudio."""
    try:
        seletores_audio = [
            'button[aria-label*="Reproduzir"]',
            'button[aria-label*="Play"]',
            'button[aria-label*="áudio"]',
            'button[aria-label*="voz"]',
            '[data-testid="audio-play"]',
            '[data-testid="voice-note-play"]',
            '[data-icon="audio-play"]',
            '[data-icon="ptt-play"]',
            '[data-testid="audio-waveform"]',
            'audio'
        ]
        if await mensagem.locator(", ".join(seletores_audio)).count() > 0:
            return True

        texto_bruto = (await mensagem.inner_text()).strip()
        linhas = [l.strip() for l in texto_bruto.split("\n") if l.strip()]
        for l in linhas:
            if re.match(r"^\d{1,2}:\d{2}$", l) and len(linhas) <= 3:
                return True
    except Exception:
        pass
    return False


async def mensagem_eh_recebida(mensagem) -> Optional[bool]:
    """Identifica se a mensagem veio do contato (True) ou se foi enviada por nós (False)."""
    try:
        atual = mensagem
        for _ in range(4):
            atual = atual.locator("xpath=..")
        classe = (await atual.get_attribute("class")) or ""

        if "x1ew7x2d" in classe or "message-in" in classe:
            return True
        if "x1g5lz36" in classe or "message-out" in classe:
            return False

        parent_box = mensagem.locator(
            "xpath=ancestor::div[contains(@class, 'message-in') or contains(@class, 'message-out')]"
        ).first
        if await parent_box.count() > 0:
            box_class = (await parent_box.get_attribute("class")) or ""
            if "message-in" in box_class:
                return True
            if "message-out" in box_class:
                return False
    except Exception:
        pass
    return None


async def obter_texto_mensagem(mensagem) -> str:
    """Extrai o texto legível da mensagem ou identifica se é mensagem de voz/áudio."""
    try:
        if await eh_mensagem_de_audio(mensagem):
            texto_bruto = (await mensagem.inner_text()).strip()
            duracao_match = re.search(r"\b\d{1,2}:\d{2}\b", texto_bruto)
            duracao = f" ({duracao_match.group(0)})" if duracao_match else ""
            return f"[Mensagem de Áudio{duracao}]"

        elemento = mensagem.locator('[data-testid="selectable-text"]').first
        if await elemento.count() > 0:
            texto = (await elemento.inner_text()).strip()
            if texto:
                if re.match(r"^\d{1,2}:\d{2}$", texto):
                    return f"[Mensagem de Áudio ({texto})]"
                return texto
        text = (await mensagem.inner_text()).strip()
        if text:
            return text
        if await mensagem.locator('img, video, [data-icon="image"], [data-icon="video"], [data-icon="document"]').count():
            return "[Mídia]"
        return ""
    except Exception:
        return ""


async def obter_mensagem_id(mensagem) -> Optional[str]:
    """Gera um identificador único para a mensagem (metadados + texto)."""
    try:
        element_id = await mensagem.get_attribute("data-id")
        if not element_id:
            parent = mensagem.locator('xpath=ancestor::*[@data-id][1]')
            if await parent.count():
                element_id = await parent.get_attribute("data-id")
        if element_id:
            return element_id
        texto = await obter_texto_mensagem(mensagem)
        metadata = await mensagem.get_attribute("data-pre-plain-text")
        if not texto and not metadata:
            return None
        return f"{metadata or ''}|{texto}"
    except Exception:
        return None


async def obter_nome_chat_ativo(page) -> str:
    """Obtém o nome do contato ou grupo atualmente aberto no cabeçalho."""
    try:
        header = page.locator("#main header").first
        if await header.count() > 0:
            titulo_elem = header.locator('span[title], div[role="button"] span[dir="auto"]').first
            if await titulo_elem.count() > 0:
                nome = (await titulo_elem.get_attribute("title")) or (await titulo_elem.inner_text()).strip()
                if nome:
                    return nome
    except Exception:
        pass
    return "Conversa Atual"


async def abrir_chat_alvo_visivel(page) -> bool:
    """Abre apenas a linha lateral que corresponde exatamente ao contato alvo."""
    if chat_corresponde_alvo(await obter_nome_chat_ativo(page)):
        return True
    rows = page.locator('#pane-side div[role="listitem"], #pane-side div[role="row"]')
    for index in range(await rows.count()):
        row = rows.nth(index)
        if not await row.is_visible():
            continue
        label = row.locator('span[title], span[dir="auto"]').first
        if not await label.count():
            continue
        nome = ((await label.get_attribute("title")) or (await label.inner_text()) or "").strip()
        if not chat_corresponde_alvo(nome) or (IGNORAR_GRUPOS and await chat_eh_grupo(page, chat_row=row)):
            continue
        await row.click()
        await asyncio.sleep(0.4)
        return chat_corresponde_alvo(await obter_nome_chat_ativo(page)) and not (
            IGNORAR_GRUPOS and await chat_eh_grupo(page))
    return False


async def obter_contexto_conversa(page) -> List[Dict[str, str]]:
    """Lê o histórico recente da conversa aberta para enriquecer o contexto da IA."""
    contexto = []
    try:
        mensagens = page.locator('#main [data-pre-plain-text]')
        total = await mensagens.count()
        inicio = max(0, total - LIMITE_CONTEXTO)

        for i in range(inicio, total):
            msg = mensagens.nth(i)
            texto = await obter_texto_mensagem(msg)
            if not texto:
                continue

            recebida = await mensagem_eh_recebida(msg)
            if recebida is True:
                contexto.append({"role": "user", "content": texto})
            elif recebida is False:
                contexto.append({"role": "assistant", "content": texto})
    except Exception:
        pass
    return contexto



async def enviar_mensagem(page, texto: str) -> bool:
    """Digita e envia uma mensagem no WhatsApp."""
    texto = texto.strip()
    if not texto:
        return False

    seletores = [
        'div[contenteditable="true"][data-tab="10"]',
        'footer div[contenteditable="true"]',
        'div[contenteditable="true"][role="textbox"]'
    ]

    campo = None
    for sel in seletores:
        candidato = page.locator(sel).last
        if (await candidato.count() > 0) and (await candidato.is_visible()):
            campo = candidato
            break

    if not campo:
        print("⚠️ Caixa de digitação do WhatsApp não encontrada.")
        return False

    await campo.click()
    await campo.fill(texto)
    await campo.press("Enter")
    for _ in range(8):
        await asyncio.sleep(0.25)
        ultima = page.locator('#main [data-pre-plain-text]').last
        if await ultima.count() and await mensagem_eh_recebida(ultima) is False:
            if (await obter_texto_mensagem(ultima)).strip() == texto:
                return True
    return False


async def abrir_chat_por_nome(page, nome_contato: str) -> bool:
    """
    Localiza e abre o chat pelo nome na lista lateral ou pela barra de busca.
    """
    if not nome_contato:
        return False

    try:
        # 1. Tenta clicar diretamente na linha do contato na barra lateral (#pane-side)
        chat_item = page.locator('#pane-side').locator(f'span[title="{nome_contato}"]').first
        if await chat_item.count() > 0 and await chat_item.is_visible():
            await chat_item.click()
            await asyncio.sleep(0.5)
            return True

        # Tenta localizar por texto na linha do chat
        chat_by_text = page.locator('#pane-side div[role="listitem"], #pane-side div[role="row"]').filter(has_text=nome_contato).first
        if await chat_by_text.count() > 0 and await chat_by_text.is_visible():
            await chat_by_text.click()
            await asyncio.sleep(0.5)
            return True

        # 2. Se não estiver visível diretamente, pesquisa na barra de busca do WhatsApp
        seletores_busca = [
            '[data-testid="chat-list-search"]',
            '#side div[contenteditable="true"]',
            'div[contenteditable="true"][data-tab="3"]',
            'div[contenteditable="true"][role="textbox"]'
        ]

        search_box = None
        for sel in seletores_busca:
            candidato = page.locator(sel).first
            if await candidato.count() > 0 and await candidato.is_visible():
                search_box = candidato
                break

        if search_box:
            await search_box.click()
            await search_box.fill(nome_contato)
            await asyncio.sleep(0.8)

            # Procura o resultado da busca
            resultado = page.locator('#pane-side').locator(f'span[title="{nome_contato}"]').first
            if await resultado.count() == 0:
                resultado = page.locator('#pane-side div[role="listitem"]').filter(has_text=nome_contato).first

            if await resultado.count() > 0:
                await resultado.click()
                await asyncio.sleep(0.6)
                # Limpa a busca
                await page.keyboard.press("Escape")
                return True
            else:
                await page.keyboard.press("Escape")
    except Exception as erro:
        print(f"❌ Erro ao tentar abrir chat '{nome_contato}': {erro}")
    return False



async def fechar_chat_ativo(page):
    """
    Deseleciona/fecha a conversa atualmente aberta no WhatsApp Web,
    retornando à tela inicial para permitir que novas mensagens gerem badges na barra lateral.
    """
    try:
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.3)
    except Exception:
        pass


async def chat_eh_grupo(page, chat_row=None) -> bool:
    """
    Verifica se o chat atual (ou linha do chat) é um grupo ou comunidade.
    Utiliza:
    1. Ícones e marcadores de grupo/comunidade na lista lateral e no cabeçalho do CHAT ATIVO (#main header).
    2. Presença de múltiplos remetentes distintos nas mensagens do chat aberto (#main).
    3. Descrição/Subtítulo no cabeçalho (#main header) (ex: participantes, dados do grupo).
    """
    try:
        # 1. Se fornecida a linha da lista lateral (#pane-side)
        if chat_row:
            seletores_lateral = [
                '[data-icon="default-group"]',
                '[data-icon="community"]',
                '[data-icon="community-outline"]',
                '[data-icon="newsletter"]',
                '[data-testid="default-group"]',
                '[data-testid="community-icon"]',
                'span[data-icon*="group"]',
                'span[data-icon*="community"]'
            ]
            for sel in seletores_lateral:
                if await chat_row.locator(sel).count() > 0:
                    return True

            avatar = chat_row.locator('div[role="img"]').first
            if await avatar.count() > 0:
                rotulo = (await avatar.get_attribute("aria-label")) or ""
                if any(termo in rotulo.lower() for termo in ["grupo", "group", "comunidade", "community"]):
                    return True
            # A linha lateral pertence ao candidato; o cabeçalho e as mensagens
            # ainda podem pertencer a outra conversa aberta.
            return False

        # 2. Verificação no cabeçalho do CHAT ATIVO (#main header)
        # Importante: usar '#main header' para NÃO confundir com o cabeçalho da barra lateral (#side header)!
        main_header = page.locator("#main header").first
        if await main_header.count() > 0:
            seletores_header_icone = [
                '[data-icon="default-group"]',
                '[data-icon="community"]',
                '[data-testid="default-group"]',
                '[data-testid="community-icon"]'
            ]
            for sel in seletores_header_icone:
                if await main_header.locator(sel).count() > 0:
                    return True

            elementos_sub = main_header.locator('span[title], span[dir="auto"], div[role="button"] + span')
            for i in range(await elementos_sub.count()):
                elem = elementos_sub.nth(i)
                texto = ((await elem.get_attribute("title")) or (await elem.inner_text()) or "").strip().lower()
                if "," in texto or any(kw in texto for kw in [
                    "clique aqui para dados do grupo",
                    "click here for group info",
                    "admin",
                    "você, ",
                    "you, ",
                    "participantes",
                    "participants"
                ]):
                    return True

        # 3. Verificação por Múltiplos Remetentes distintos nas mensagens do chat aberto (#main)
        mensagens = page.locator('#main [data-pre-plain-text]')
        total_msgs = await mensagens.count()
        if total_msgs > 0:
            remetentes_recebidos = set()
            inicio = max(0, total_msgs - 10)
            for i in range(inicio, total_msgs):
                msg = mensagens.nth(i)
                if await mensagem_eh_recebida(msg) is True:
                    pre_text = (await msg.get_attribute("data-pre-plain-text")) or ""
                    match = re.search(r"\]\s*([^:]+):", pre_text)
                    if match:
                        remetente = match.group(1).strip()
                        remetentes_recebidos.add(remetente)

            # Mais de um remetente em mensagens recebidas -> Grupo!
            if len(remetentes_recebidos) > 1:
                return True

    except Exception:
        pass

    return False



# ============================================================
# DETECÇÃO DE CHATS NÃO LIDOS
# ============================================================

async def encontrar_chats_nao_lidos(page, chats_ignorados: set = None) -> List[Dict[str, Any]]:
    """Varre #pane-side procurando por conversas com badge de não lidas."""
    chats_nao_lidos = []
    nomes_filtros_ignorados = {
        "tudo", "todas", "não lidas", "nao lidas", "favoritas", "favoritos",
        "grupos", "all", "unread", "favorites", "groups", "conversas", "chats"
    }

    try:
        pane = page.locator("#pane-side")
        if await pane.count() == 0:
            return chats_nao_lidos

        chat_rows = pane.locator('div[role="listitem"], div[role="row"]')
        total_rows = await chat_rows.count()

        seletores_badge = [
            'span[aria-label*="não lida"]',
            'span[aria-label*="não lidas"]',
            'span[aria-label*="nao lida"]',
            'span[aria-label*="unread"]',
            '[data-testid="unread-count"]',
            '[data-testid="icon-unread-count"]',
            'span[aria-label*="mensagem não lida"]',
            'span[aria-label*="mensagens não lidas"]'
        ]
        badge_query = ", ".join(seletores_badge)

        for i in range(total_rows):
            row = chat_rows.nth(i)
            try:
                if not (await row.is_visible()):
                    continue

                role = (await row.get_attribute("role")) or ""
                if role == "tab" or (await row.locator('[role="tab"]').count() > 0):
                    continue

                badge = row.locator(badge_query).first
                if (await badge.count() == 0) or not (await badge.is_visible()):
                    continue

                span_nome = row.locator('span[title], span[dir="auto"]').first
                nome_contato = ""
                if await span_nome.count() > 0:
                    nome_contato = ((await span_nome.get_attribute("title")) or (await span_nome.inner_text())).strip()

                if not nome_contato or nome_contato.lower() in nomes_filtros_ignorados:
                    continue

                if chats_ignorados and nome_contato in chats_ignorados:
                    continue

                # Se a opção IGNORAR_GRUPOS estiver ativa e o chat for grupo, pula
                if IGNORAR_GRUPOS and await chat_eh_grupo(page, chat_row=row):
                    continue

                chats_nao_lidos.append({
                    "nome": nome_contato,
                    "elemento": row,
                    "quantidade": int((re.search(r"\d+", (await badge.inner_text()) or "") or ["1"])[0]),
                })
            except Exception:
                continue

    except Exception:
        pass

    return chats_nao_lidos


async def diagnostico_scanner(page) -> str:
    """Contagens do DOM sem nome de contato nem texto de mensagens."""
    pane = page.locator("#pane-side")
    rows = pane.locator('div[role="listitem"], div[role="row"]')
    badges = pane.locator(
        'span[aria-label*="não lida"], span[aria-label*="nao lida"], '
        'span[aria-label*="unread"], [data-testid*="unread"]'
    )
    messages = page.locator('#main [data-pre-plain-text]')
    message_count = await messages.count()
    direction = await mensagem_eh_recebida(messages.last) if message_count else None
    active_name = await obter_nome_chat_ativo(page)
    return (f"linhas_laterais={await rows.count()}, badges={await badges.count()}, "
            f"chat_aberto={active_name != 'Conversa Atual'}, mensagens_visiveis={message_count}, "
            f"ultima_recebida={direction}")


# ============================================================
# TRÊS CAMADAS ASSÍNCRONAS: SCANNER, POOL DE IA E DESPACHANTE
# ============================================================

async def scanner_chat_unico(page, fila_entrada: asyncio.Queue, lock_pagina: asyncio.Lock,
                             mensagens_processadas: set, chats_em_processamento: set):
    """Observa somente mensagens novas no chat alvo já aberto pelo operador."""
    ultimo_id = None
    ultimo_pulso = 0.0
    ultima_tentativa_abertura = 0.0
    print("👀 [Scanner] Modo chat único: procurando apenas o contato configurado.")
    while True:
        try:
            async with lock_pagina:
                nome_aberto = await obter_nome_chat_ativo(page)
                corresponde = chat_corresponde_alvo(nome_aberto)
                if not corresponde and time.monotonic() - ultima_tentativa_abertura >= 10:
                    ultima_tentativa_abertura = time.monotonic()
                    if await abrir_chat_alvo_visivel(page):
                        print("🔗 [Scanner] Contato alvo aberto e conferido no cabeçalho.")
                        nome_aberto = await obter_nome_chat_ativo(page)
                        corresponde = chat_corresponde_alvo(nome_aberto)
                grupo = await chat_eh_grupo(page) if corresponde and IGNORAR_GRUPOS else False
                if time.time() - ultimo_pulso >= 15:
                    print(f"💓 [Scanner] chat_aberto={nome_aberto != 'Conversa Atual'}, "
                          f"alvo_confere={corresponde}, grupo={grupo}")
                    ultimo_pulso = time.time()
                if not corresponde:
                    ultimo_id = None
                elif grupo:
                    ultimo_id = None
                else:
                    mensagens = page.locator('#main [data-pre-plain-text]')
                    total = await mensagens.count()
                    ids = [await obter_mensagem_id(mensagens.nth(i)) for i in range(total)]
                    if ids:
                        if ultimo_id is None or ultimo_id not in ids:
                            ultimo_id = ids[-1]
                            print("📌 [Scanner] Chat alvo observado; histórico visível ignorado.")
                        else:
                            inicio = ids.index(ultimo_id) + 1
                            for index in range(inicio, total):
                                if fila_entrada.full():
                                    break
                                msg_id = ids[index]
                                if not msg_id:
                                    continue
                                key = f"{CHAT_ALVO}:{msg_id}"
                                if key in mensagens_processadas or delivery_state(delivery_key(CHAT_ALVO, msg_id)):
                                    ultimo_id = msg_id
                                    continue
                                candidate = mensagens.nth(index)
                                if await mensagem_eh_recebida(candidate) is not True:
                                    ultimo_id = msg_id
                                    continue
                                texto = await obter_texto_mensagem(candidate)
                                if not texto:
                                    ultimo_id = msg_id
                                    continue
                                metadata = (await candidate.get_attribute("data-pre-plain-text")) or ""
                                contexto = await obter_contexto_conversa(page)
                                if not chat_corresponde_alvo(await obter_nome_chat_ativo(page)):
                                    ultimo_id = None
                                    break
                                fila_entrada.put_nowait(MensagemTarefa(CHAT_ALVO, msg_id, texto, metadata, contexto))
                                mensagens_processadas.add(key)
                                chats_em_processamento.add(CHAT_ALVO)
                                save_message(CHAT_ALVO, delivery_key(CHAT_ALVO, msg_id), "user", "[conteúdo omitido]")
                                ultimo_id = msg_id
                                print(f"📥 [Scanner] Mensagem nova no chat alvo; fila={fila_entrada.qsize()}")
            await asyncio.sleep(INTERVALO_SCANNER)
        except asyncio.CancelledError:
            break
        except Exception as erro:
            print(f"⚠️ [Scanner] Erro no chat único: {erro}")
            await asyncio.sleep(1.5)

async def scanner_loop(
    page,
    fila_entrada: asyncio.Queue,
    lock_pagina: asyncio.Lock,
    mensagens_processadas: set,
    chats_em_processamento: set,
    chats_ignorados: set,
    baseline_unread_counts: dict[str, int]
):
    """
    PRODUTOR: Varre o WhatsApp, abre chats não lidos rapidamente,
    e também monitora o chat atualmente aberto na tela caso receba mensagens novas ao vivo.
    """
    print("👀 [Scanner] Monitor de mensagens ativado (varrendo lista lateral + chat ativo)...")
    ultimo_pulso = time.time()
    chats_pendentes: set[str] = set()
    ultimo_id_chat_aberto: dict[str, str] = {}

    while True:
        try:
            # Pulso periódico no console a cada 15 segundos para confirmar que o scanner está vivo
            if time.time() - ultimo_pulso > 15:
                async with lock_pagina:
                    diagnostico = await diagnostico_scanner(page)
                print(f"💓 [Scanner] Em espera... {diagnostico}")
                ultimo_pulso = time.time()

            # 1. Busca chats não lidos na lista lateral
            chats_nao_lidos = []
            async with lock_pagina:
                chats_nao_lidos = await encontrar_chats_nao_lidos(page, chats_ignorados)
            nomes_encontrados = {item["nome"] for item in chats_nao_lidos}
            for name in list(baseline_unread_counts):
                if name not in nomes_encontrados:
                    del baseline_unread_counts[name]
            novos_chats = []
            for item in chats_nao_lidos:
                name = item["nome"]
                previous = baseline_unread_counts.get(name, 0)
                current = item["quantidade"]
                if current > previous:
                    item["quantidade"] = current - previous
                    novos_chats.append(item)
                baseline_unread_counts[name] = current
            chats_nao_lidos = novos_chats
            chats_nao_lidos.extend(
                {"nome": name, "elemento": None, "quantidade": 1}
                for name in sorted(chats_pendentes - {item["nome"] for item in chats_nao_lidos})
            )

            # 2. Se houver conversas não lidas na lista lateral, abre e processa
            if chats_nao_lidos:
                for chat_info in chats_nao_lidos:
                    nome = chat_info["nome"]

                    if nome in chats_ignorados:
                        continue
                    if fila_entrada.full():
                        async with lock_pagina:
                            active_name = await obter_nome_chat_ativo(page)
                        if active_name and active_name != "Conversa Atual":
                            chats_pendentes.add(active_name)
                        break

                    async with lock_pagina:
                        try:
                            if chat_info["elemento"] is None:
                                if not await abrir_chat_por_nome(page, nome):
                                    continue
                            else:
                                await chat_info["elemento"].click()
                            await asyncio.sleep(0.4)
                        except Exception:
                            continue

                        active_name = await obter_nome_chat_ativo(page)
                        if active_name != nome:
                            print("⚠️ [Scanner] Chat aberto não corresponde à linha escolhida; mensagem ignorada.")
                            continue

                        # Se for grupo, ignora
                        if IGNORAR_GRUPOS and await chat_eh_grupo(page):
                            print(f"👥 [Scanner] Grupo detectado e ignorado: '{nome}'")
                            chats_ignorados.add(nome)
                            continue

                        mensagens = page.locator('#main [data-pre-plain-text]')
                        total = await mensagens.count()
                        if total == 0:
                            continue

                        contexto = await obter_contexto_conversa(page)
                        ids = [await obter_mensagem_id(mensagens.nth(i)) for i in range(total)]
                        if ids[-1]:
                            ultimo_id_chat_aberto[nome] = ids[-1]
                        seen = [i for i, msg_id in enumerate(ids)
                                if msg_id and f"{nome}:{msg_id}" in mensagens_processadas]
                        restantes = max(1, chat_info["quantidade"])
                        for index in range(total - 1, -1, -1):
                            candidate = mensagens.nth(index)
                            if await mensagem_eh_recebida(candidate) is True:
                                restantes -= 1
                            if restantes <= 0:
                                inicio = index
                                break
                        else:
                            inicio = total - 1
                        if seen:
                            inicio = seen[-1] + 1
                        chats_pendentes.discard(nome)
                        for index in range(inicio, total):
                            if fila_entrada.full():
                                print("⚠️ Fila de análise cheia; mensagens restantes aguardam nova varredura.")
                                chats_pendentes.add(nome)
                                break
                            candidate = mensagens.nth(index)
                            msg_id = ids[index]
                            key = f"{nome}:{msg_id}"
                            if not msg_id or key in mensagens_processadas:
                                continue
                            if delivery_state(delivery_key(nome, msg_id)):
                                mensagens_processadas.add(key)
                                continue
                            if await mensagem_eh_recebida(candidate) is not True:
                                continue
                            texto = await obter_texto_mensagem(candidate)
                            if not texto:
                                continue
                            metadata = (await candidate.get_attribute("data-pre-plain-text")) or ""
                            if await obter_nome_chat_ativo(page) != nome:
                                print("⚠️ [Scanner] Chat mudou durante a leitura; lote descartado.")
                                break
                            fila_entrada.put_nowait(MensagemTarefa(nome, msg_id, texto, metadata, contexto))
                            mensagens_processadas.add(key)
                            chats_em_processamento.add(nome)
                            save_message(nome, delivery_key(nome, msg_id), "user", "[conteúdo omitido]")
                            print(f"📥 [Scanner] '{nome}': mensagem enfileirada; fila={fila_entrada.qsize()}")
            else:
                # 3. Se não houver chats com bolinha verde, monitora o chat que já está ABERTO na tela
                async with lock_pagina:
                    nome_ativo = await obter_nome_chat_ativo(page)
                    if nome_ativo and nome_ativo != "Conversa Atual":
                        if nome_ativo in chats_ignorados:
                            pass
                        elif IGNORAR_GRUPOS and await chat_eh_grupo(page):
                            chats_ignorados.add(nome_ativo)
                            print(f"👥 [Scanner] Chat aberto é grupo ignorado: '{nome_ativo}'")
                        else:
                            mensagens = page.locator('#main [data-pre-plain-text]')
                            total = await mensagens.count()
                            if total > 0:
                                ids = [await obter_mensagem_id(mensagens.nth(i)) for i in range(total)]
                                ultimo_id = ultimo_id_chat_aberto.get(nome_ativo)
                                if ultimo_id is None or ultimo_id not in ids:
                                    # Ao abrir/trocar de chat, as mensagens visíveis são histórico.
                                    # Só as que surgirem depois desta observação são novas.
                                    if ids[-1]:
                                        ultimo_id_chat_aberto[nome_ativo] = ids[-1]
                                    await asyncio.sleep(INTERVALO_SCANNER)
                                    continue
                                inicio = ids.index(ultimo_id) + 1
                                contexto = await obter_contexto_conversa(page)
                                for index in range(inicio, total):
                                    if fila_entrada.full():
                                        print("⚠️ Fila de análise cheia; nova varredura tentará mensagens pendentes.")
                                        chats_pendentes.add(nome_ativo)
                                        break
                                    msg_id = ids[index]
                                    key = f"{nome_ativo}:{msg_id}"
                                    if not msg_id or key in mensagens_processadas:
                                        continue
                                    if delivery_state(delivery_key(nome_ativo, msg_id)):
                                        mensagens_processadas.add(key)
                                        continue
                                    candidate = mensagens.nth(index)
                                    if await mensagem_eh_recebida(candidate) is not True:
                                        continue
                                    texto = await obter_texto_mensagem(candidate)
                                    if not texto:
                                        continue
                                    metadata = (await candidate.get_attribute("data-pre-plain-text")) or ""
                                    if await obter_nome_chat_ativo(page) != nome_ativo:
                                        print("⚠️ [Scanner] Chat mudou durante a leitura; lote descartado.")
                                        break
                                    fila_entrada.put_nowait(MensagemTarefa(nome_ativo, msg_id, texto, metadata, contexto))
                                    mensagens_processadas.add(key)
                                    chats_em_processamento.add(nome_ativo)
                                    save_message(nome_ativo, delivery_key(nome_ativo, msg_id), "user", "[conteúdo omitido]")
                                    print(f"📥 [Scanner] '{nome_ativo}': mensagem enfileirada; fila={fila_entrada.qsize()}")
                                if ids[-1] and nome_ativo not in chats_pendentes:
                                    ultimo_id_chat_aberto[nome_ativo] = ids[-1]



            await asyncio.sleep(INTERVALO_SCANNER)

        except asyncio.CancelledError:
            break
        except Exception as erro:
            print(f"⚠️ [Scanner] Erro na varredura: {erro}")
            await asyncio.sleep(1.5)


async def worker_ia_loop(
    worker_id: int,
    fila_entrada: asyncio.Queue,
    fila_saida: asyncio.Queue
):
    """
    WORKER / AGENTE DE IA: Puxa tarefas da fila_entrada e gera
    respostas pela ponte RAG local, sem fallback para outro modelo.
    """
    print(f"🤖 [Worker IA {worker_id}] Pronto e aguardando tarefas...")

    async def relatar_espera(inicio: float):
        while True:
            await asyncio.sleep(15)
            print(f"⏳ [Worker IA {worker_id}] Análise RAG em andamento há {time.monotonic() - inicio:.0f}s...")

    while True:
        progresso = None
        try:
            tarefa: MensagemTarefa = await fila_entrada.get()

            print(f"🧠 [Worker IA {worker_id}] Processando resposta para '{tarefa.chat_nome}'...")
            t_inicio = time.monotonic()

            try:
                if tarefa.texto.startswith("[Mensagem de Áudio"):
                    resposta = "Ainda não consigo ouvir áudio. Pode enviar a mensagem por texto?"
                elif tarefa.texto.startswith("[Mídia]"):
                    resposta = "Ainda não consigo analisar essa mídia. Pode descrever o conteúdo em texto?"
                else:
                    contexto = []
                    for item in tarefa.contexto[-4:]:
                        if item.get("role") == "user":
                            contexto.append({"pergunta": str(item.get("content", ""))[:500], "resposta": ""})
                        elif contexto and item.get("role") == "assistant":
                            contexto[-1]["resposta"] = str(item.get("content", ""))[:500]
                    request_id = delivery_key(tarefa.chat_nome, tarefa.mensagem_id)
                    progresso = asyncio.create_task(relatar_espera(t_inicio))
                    answer = await analyze(request_id, tarefa.texto, contexto[-2:])
                    resposta = format_answer(answer)
                duracao = time.monotonic() - t_inicio

                print(f"✨ [Worker IA {worker_id}] Resposta gerada para '{tarefa.chat_nome}' em {duracao:.2f}s!")

                resp_tarefa = RespostaTarefa(
                    chat_nome=tarefa.chat_nome,
                    mensagem_id=tarefa.mensagem_id,
                    resposta_texto=resposta
                )
                await fila_saida.put(resp_tarefa)

            except Exception as erro:
                print(f"❌ [Worker IA {worker_id}] Análise encerrada em {time.monotonic() - t_inicio:.1f}s: "
                      f"{type(erro).__name__}: {erro}")
                aviso = (
                    "Encontrei dados pessoais na mensagem. Remova telefone, e-mail, documento "
                    "ou credenciais e envie novamente o trecho que deseja verificar."
                    if isinstance(erro, BridgeError) and erro.code == "PERSONAL_DATA"
                    else "Não consegui analisar esta mensagem agora. Tente novamente mais tarde."
                )
                await fila_saida.put(RespostaTarefa(
                    chat_nome=tarefa.chat_nome,
                    mensagem_id=tarefa.mensagem_id,
                    resposta_texto=aviso,
                ))

            finally:
                if progresso is not None:
                    progresso.cancel()
                fila_entrada.task_done()

        except asyncio.CancelledError:
            break
        except Exception as erro:
            print(f"⚠️ [Worker IA {worker_id}] Erro inesperado: {erro}")
            await asyncio.sleep(1.0)


async def dispatcher_loop(
    page,
    fila_saida: asyncio.Queue,
    lock_pagina: asyncio.Lock,
    chats_em_processamento: set
):
    """
    CONSUMIDOR / DESPACHANTE: Puxa respostas prontas da fila_saida,
    localiza o chat correspondente no WhatsApp e realiza o envio seguro.
    """
    print("📤 [Despachante] Inicializado e aguardando respostas prontas...")

    while True:
        resposta_tarefa = None
        try:
            resposta_tarefa: RespostaTarefa = await fila_saida.get()

            print("\n" + "═" * 70)
            print(f"🚀 [Despachante] ENVIANDO RESPOSTA PRONTA")
            print(f"👤 Destinatário: {resposta_tarefa.chat_nome}")
            print(f"💬 Resposta preparada ({len(resposta_tarefa.resposta_texto)} caracteres)")
            print("═" * 70)

            sent = False
            if MODO_DRY_RUN or resposta_tarefa.chat_nome not in CONTATOS_AUTORIZADOS:
                print(f"💬 [Prévia dry-run]\n{resposta_tarefa.resposta_texto}")
                print("🔒 [MODO SIMULAÇÃO] Resposta NÃO enviada ao WhatsApp.")
                if not MODO_CHAT_UNICO:
                    async with lock_pagina:
                        await fechar_chat_ativo(page)
            else:
                async with lock_pagina:
                    # Verifica se o chat já está aberto ou precisa ser aberto
                    nome_atual = await obter_nome_chat_ativo(page)
                    if not (chat_corresponde_alvo(nome_atual) if MODO_CHAT_UNICO else nome_atual == resposta_tarefa.chat_nome):
                        if MODO_CHAT_UNICO:
                            raise RuntimeError("Chat alvo não está aberto; envio cancelado")
                        print(f"🔄 [Despachante] Navegando para o chat de '{resposta_tarefa.chat_nome}'...")
                        sucesso = await abrir_chat_por_nome(page, resposta_tarefa.chat_nome)
                        if not sucesso:
                            raise RuntimeError("Chat de destino indisponível; resposta não reenviada automaticamente")

                    nome_final = await obter_nome_chat_ativo(page)
                    if not (chat_corresponde_alvo(nome_final) if MODO_CHAT_UNICO else nome_final == resposta_tarefa.chat_nome):
                        raise RuntimeError("Identidade do chat mudou antes do envio")
                    key = delivery_key(resposta_tarefa.chat_nome, resposta_tarefa.mensagem_id)
                    if not reserve_delivery(key):
                        raise RuntimeError("Envio já tentado ou reserva indisponível; conferir manualmente")
                    sent = await enviar_mensagem(page, resposta_tarefa.resposta_texto)
                    if not sent:
                        raise RuntimeError("Não houve confirmação do envio no WhatsApp")
                    if not confirm_delivery(key):
                        raise RuntimeError("Envio visto no WhatsApp, mas confirmação local falhou")
                    print(f"✅ [Despachante] Mensagem enviada para '{resposta_tarefa.chat_nome}'!")

                    # Fecha o chat ativo para retornar à tela inicial e permitir que novas mensagens gerem badges
                    if not MODO_CHAT_UNICO:
                        await fechar_chat_ativo(page)

            if sent:
                asst_msg_id = "assistant_" + delivery_key(
                    resposta_tarefa.chat_nome, resposta_tarefa.mensagem_id
                )
                save_message(
                    chat_nome=resposta_tarefa.chat_nome,
                    mensagem_id=asst_msg_id,
                    role="assistant",
                    texto="[resposta enviada]",
                )

            # Libera o contato para receber novas perguntas
            chats_em_processamento.discard(resposta_tarefa.chat_nome)
            fila_saida.task_done()


        except asyncio.CancelledError:
            break
        except Exception as erro:
            print(f"⚠️ [Despachante] Erro no envio: {erro}")
            if resposta_tarefa is not None:
                chats_em_processamento.discard(resposta_tarefa.chat_nome)
                fila_saida.task_done()
            await asyncio.sleep(1.0)


# ============================================================
# FLUXO PRINCIPAL DE EXECUÇÃO
# ============================================================

async def main_async():
    print("=" * 70)
    print("🤖 BOT WHATSAPP MULTI-AGENTES & SISTEMA DE FILAS ASSÍNCRONAS")
    print(f"⚙️  Modo Atual: {'🔒 SIMULAÇÃO / DRY-RUN (Apenas Console)' if MODO_DRY_RUN else '🚀 PRODUÇÃO (Envio Real Ativado)'}")
    print("🧠 RAG local via ponte HTTP; Ollama/Qwen configurado no repositório principal")
    print(f"⚡ Workers de IA Concorrentes: {NUM_WORKERS_IA}")
    print("=" * 70)
    print()

    # Filas e estruturas de sincronização
    if not init_database():
        raise RuntimeError("Não foi possível inicializar o banco local")
    if not await health():
        raise RuntimeError("Ponte RAG indisponível. Inicie-a e confirme /health antes do bot")
    if not MODO_DRY_RUN and not CONTATOS_AUTORIZADOS:
        raise RuntimeError("Defina WHATSAPP_ALLOWED_CONTACTS para habilitar envio")
    if MODO_CHAT_UNICO and not CHAT_ALVO:
        raise RuntimeError("Defina BOT_TARGET_CHAT com o nome exato do contato para o modo chat único")
    if MODO_CHAT_UNICO and not MODO_DRY_RUN and CHAT_ALVO not in CONTATOS_AUTORIZADOS:
        raise RuntimeError("BOT_TARGET_CHAT deve constar em WHATSAPP_ALLOWED_CONTACTS para envio")
    fila_entrada = asyncio.Queue(maxsize=FILA_MAXIMA)
    fila_saida = asyncio.Queue(maxsize=FILA_MAXIMA)
    lock_pagina = asyncio.Lock()
    mensagens_processadas = set()
    chats_em_processamento = set()
    chats_ignorados = set()

    async with async_playwright() as p:
        print("🌐 Inicializando Chromium...")
        context = await p.chromium.launch_persistent_context(
            PERFIL_WHATSAPP,
            headless=False,
            viewport={"width": 1280, "height": 850}
        )

        page = context.pages[0] if context.pages else await context.new_page()

        print("🔗 Acessando WhatsApp Web...")
        await page.goto("https://web.whatsapp.com")

        print("⏳ Aguardando carregamento da interface do WhatsApp...")
        try:
            await page.wait_for_selector("#pane-side", timeout=60000)
            print("✅ WhatsApp Web conectado e carregado com sucesso!")
        except Exception:
            print("\n⚠️ O painel lateral ainda não foi detectado.")
            print("👉 Se necessário, escaneie o QR Code no navegador.")
            await asyncio.to_thread(input, "👉 Pressione ENTER quando o WhatsApp estiver logado...")

        # Mapeia mensagens antigas do chat aberto, mas deixa a última para ser processada se for recebida
        try:
            mensagens_iniciais = page.locator('#main [data-pre-plain-text]')
            total_iniciais = await mensagens_iniciais.count()
            nome_inicial = await obter_nome_chat_ativo(page)
            # Sem opt-in, o bot só deve analisar mensagens recebidas após iniciar.
            limite_inicial = max(0, total_iniciais - 1) if PROCESSAR_NAO_LIDAS_ANTIGAS else total_iniciais
            for i in range(limite_inicial):
                msg_id = await obter_mensagem_id(mensagens_iniciais.nth(i))
                if msg_id:
                    mensagens_processadas.add(f"{nome_inicial}:{msg_id}")
            print(f"📚 {len(mensagens_processadas)} mensagens antigas catalogadas.")
        except Exception:
            pass

        baseline_unread_counts = {}
        if not MODO_CHAT_UNICO and not PROCESSAR_NAO_LIDAS_ANTIGAS:
            for item in await encontrar_chats_nao_lidos(page, chats_ignorados):
                baseline_unread_counts[item["nome"]] = item["quantidade"]
            print(f"📌 {len(baseline_unread_counts)} conversas não lidas anteriores serão ignoradas até chegar nova mensagem.")

        print("\n🚀 Iniciando pipelines assíncronos (Scanner + Workers + Despachante)...\n")

        # Inicializa o pool de tarefas concorrentes
        tarefas = []

        # 1. Scanner (Produtor)
        if MODO_CHAT_UNICO:
            tarefas.append(asyncio.create_task(scanner_chat_unico(
                page, fila_entrada, lock_pagina, mensagens_processadas, chats_em_processamento)))
        else:
            tarefas.append(asyncio.create_task(
                scanner_loop(page, fila_entrada, lock_pagina, mensagens_processadas,
                             chats_em_processamento, chats_ignorados, baseline_unread_counts)
            ))

        # 2. Pool de Workers IA
        for worker_id in range(1, NUM_WORKERS_IA + 1):
            tarefas.append(asyncio.create_task(
                worker_ia_loop(worker_id, fila_entrada, fila_saida)
            ))

        # 3. Despachante (Consumidor)
        tarefas.append(asyncio.create_task(
            dispatcher_loop(page, fila_saida, lock_pagina, chats_em_processamento)
        ))

        try:
            await asyncio.gather(*tarefas)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n🛑 Finalizando bot e cancelando tarefas assíncronas...")
            for t in tarefas:
                t.cancel()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Programa encerrado pelo usuário.")


if __name__ == "__main__":
    main()
