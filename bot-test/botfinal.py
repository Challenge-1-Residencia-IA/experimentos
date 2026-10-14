"""WhatsApp Web -> fila limitada -> RAG local -> prévia/envio autorizado."""
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langdetect import DetectorFactory, detect_langs, LangDetectException

from delivery_store import DeliveryStore, message_key
from rag_client import RagClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DetectorFactory.seed = 0
UNSUPPORTED = "No momento analiso apenas texto e links públicos em português. Envie a alegação por escrito, sem dados pessoais."
UNAVAILABLE = "Não consegui concluir esta análise agora. Isso não permite confirmar nem refutar a mensagem. Tente novamente mais tarde."


@dataclass
class Message:
    chat: str
    key: str
    text: str
    unsupported: bool = False
    target_number: int = 0


def contacts(variable):
    value = json.loads(os.getenv(variable, "[]"))
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{variable} deve ser uma lista JSON de nomes exatos")
    return list(dict.fromkeys(value))


async def active_chat(page):
    titles = page.locator("#main header span[title]")
    return (await titles.first.get_attribute("title") or "").strip() if await titles.count() else ""


async def open_chat(page, name):
    if await active_chat(page) == name:
        return True
    matches = page.locator("#pane-side").get_by_title(name, exact=True)
    if await matches.count() != 1:
        return False
    await matches.click(timeout=3000)
    for _ in range(20):
        if await active_chat(page) == name:
            # O cabeçalho pode mudar antes da lista de mensagens terminar a troca.
            await asyncio.sleep(0.5)
            return await active_chat(page) == name
        await asyncio.sleep(0.1)
    return False


async def is_group(page):
    return await page.locator('#main header [data-icon="default-group"], #main header [data-icon="community"], #main header [title*="participantes"], #main header [title*="participants"]').count() > 0


async def snapshot(page, name):
    if await active_chat(page) != name or await is_group(page):
        return None
    return await page.evaluate("""expected => {
      const main = document.querySelector('#main');
      if (main?.querySelector('header span[title]')?.getAttribute('title')?.trim() !== expected) return null;
      const rows = [...main.querySelectorAll('.message-in')];
      if ([...main.querySelectorAll('[data-id]')].some(row => row.getAttribute('data-id').includes('@g.us'))) return null;
      return rows.map(row => {
        const id = row.getAttribute('data-id') || row.closest('[data-id]')?.getAttribute('data-id') || row.querySelector('[data-id]')?.getAttribute('data-id');
        const body = row.querySelector('[data-pre-plain-text]') || row;
        const text = [...body.querySelectorAll('.selectable-text')].map(el => el.innerText).join('\\n').trim();
        const media = !!row.querySelector('audio, video, [data-icon="ptt-play"], [data-icon="audio-play"], [data-testid="audio-play"], [data-icon="document"], img[src^="blob:"]');
        return {id, text, unsupported: media || !text};
      }).filter(item => item.id);
    }""", name)


async def send_confirmed(page, name, text):
    if await active_chat(page) != name or await is_group(page):
        raise RuntimeError("chat_mismatch")
    main = page.locator("#main")
    editor = main.locator('footer div[contenteditable="true"][role="textbox"]')
    if await editor.count() != 1 or (await editor.inner_text()).strip():
        raise RuntimeError("editor_unavailable_or_has_draft")
    before = await main.locator('.message-out[data-id], .message-out [data-id], [data-id]:has(.message-out)').evaluate_all("els => els.map(el => el.getAttribute('data-id'))")
    await editor.fill(text)
    if await active_chat(page) != name:
        raise RuntimeError("chat_changed")
    await editor.press("Enter")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if await active_chat(page) != name:
            raise RuntimeError("chat_changed")
        confirmed = await page.evaluate("""({before, text}) => {
          const norm = s => s.replace(/\\s+/g, ' ').trim();
          return [...document.querySelectorAll('#main .message-out')].some(row => {
            const id = row.getAttribute('data-id') || row.closest('[data-id]')?.getAttribute('data-id') || row.querySelector('[data-id]')?.getAttribute('data-id');
            const body = [...row.querySelectorAll('.selectable-text')].map(el => el.innerText).join('\\n');
            return id && !before.includes(id) && norm(body) === norm(text)
              && !!row.querySelector('[data-icon="msg-check"], [data-icon="msg-dblcheck"]')
              && !row.querySelector('[data-icon="msg-time"], [data-icon="msg-error"]');
          });
        }""", {"before": before, "text": text})
        if confirmed:
            return
        await asyncio.sleep(0.4)
    raise RuntimeError("send_unconfirmed")


async def scanner(page, targets, incoming, lock, store):
    initialized = set()
    while True:
        for target_number, name in enumerate(targets, 1):
            if incoming.full():
                break
            try:
                async with lock:
                    if not await open_chat(page, name):
                        continue
                    messages = await snapshot(page, name)
                    if messages is None or await active_chat(page) != name:
                        continue
                    if name not in initialized:
                        for item in messages:
                            store.claim(message_key(name, item["id"]), "baseline")
                        initialized.add(name)
                        print(f"Scanner alvo {target_number}: histórico inicial ignorado; pronto para mensagem nova.")
                        continue
                    for item in messages:
                        if incoming.full():
                            break
                        key = message_key(name, item["id"])
                        if not store.known(key) and store.claim(key):
                            incoming.put_nowait(Message(name, key, item["text"], item["unsupported"], target_number))
                            print(f"Enfileirada {key[:12]}; alvo={target_number}; fila={incoming.qsize()}")
            except Exception as error:
                print(f"Scanner: falha de leitura ({type(error).__name__}); próxima varredura tentará novamente.")
        await asyncio.sleep(1)


async def worker(incoming, outgoing, client, store):
    while True:
        message = await incoming.get()
        started = time.monotonic()
        try:
            store.state(message.key, "analyzing")
            other_language = False
            if not message.unsupported and not message.text.strip().startswith(("http://", "https://")) and len(message.text.split()) >= 5:
                try:
                    language = detect_langs(message.text)[0]
                    other_language = language.lang != "pt" and language.prob >= 0.95
                except LangDetectException:
                    pass
            if message.unsupported or other_language:
                response = UNSUPPORTED
            else:
                task = asyncio.create_task(client.analyze(message.text, message.key))
                try:
                    while not task.done():
                        done, _ = await asyncio.wait({task}, timeout=15)
                        if not done:
                            print(f"RAG {message.key[:12]} em análise há {time.monotonic() - started:.0f}s")
                    response = task.result()
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            store.state(message.key, "ready")
            await outgoing.put((message, response))
            print(f"Análise {message.key[:12]} concluída em {time.monotonic() - started:.1f}s")
        except asyncio.CancelledError:
            store.state(message.key, "interrupted")
            raise
        except Exception as error:
            print(f"RAG {message.key[:12]} falhou ({type(error).__name__}).")
            store.state(message.key, "ready")
            await outgoing.put((message, UNAVAILABLE))
        finally:
            incoming.task_done()


async def dispatcher(page, outgoing, lock, store, enabled, allowed):
    while True:
        message, response = await outgoing.get()
        try:
            if not enabled:
                print(f"DRY-RUN {message.key[:12]} alvo={message.target_number} — nenhuma mensagem enviada:\n{response}\n")
                store.state(message.key, "dry_run")
                continue
            if message.chat not in allowed or len(response) > 12000:
                store.state(message.key, "blocked")
                print(f"Envio {message.key[:12]} bloqueado: contato ou tamanho não autorizado.")
                continue
            async with lock:
                if not await open_chat(page, message.chat) or await active_chat(page) != message.chat:
                    store.state(message.key, "blocked")
                    continue
                if not store.reserve_send(message.key):
                    continue
                await send_confirmed(page, message.chat, response)
                store.state(message.key, "sent")
                print(f"Envio confirmado na interface: {message.key[:12]}")
        except asyncio.CancelledError:
            store.state(message.key, "uncertain")
            raise
        except Exception as error:
            store.state(message.key, "uncertain")
            print(f"Envio {message.key[:12]} não confirmado ({type(error).__name__}); sem reenvio automático.")
        finally:
            outgoing.task_done()


async def main_async():
    targets = contacts("BOT_TARGET_CHATS")
    if not targets and os.getenv("BOT_TARGET_CHAT", "").strip():
        targets = [os.environ["BOT_TARGET_CHAT"].strip()]
    allowed = set(contacts("BOT_ALLOWED_CONTACTS"))
    enabled = os.getenv("BOT_ENABLE_SEND", "false").lower() == "true"
    if not targets or (enabled and not set(targets).issubset(allowed)):
        raise ValueError("Configure contatos alvo; envio exige todos os alvos na lista autorizada")
    size = int(os.getenv("BOT_QUEUE_SIZE", "20"))
    if not 1 <= size <= 100:
        raise ValueError("BOT_QUEUE_SIZE deve estar entre 1 e 100")
    incoming, outgoing = asyncio.Queue(size), asyncio.Queue(size)
    lock = asyncio.Lock()
    client = RagClient(os.getenv("RAG_BRIDGE_URL", "http://127.0.0.1:8765"), float(os.getenv("RAG_TIMEOUT_SECONDS", "200")))
    store, tasks = None, []
    try:
        await client.health()
        store = DeliveryStore(ROOT / "data" / "deliveries.db")
        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(str(ROOT / "whatsapp_profile"), headless=False)
            try:
                page = context.pages[0] if context.pages else await context.new_page()
                page.set_default_timeout(5000)
                await page.goto("https://web.whatsapp.com")
                print("Escaneie o QR se necessário. Aguardando WhatsApp (até 3 minutos).")
                await page.locator("#pane-side").wait_for(timeout=180000)
                print("ENVIO REAL AUTORIZADO" if enabled else "DRY-RUN: envio desligado")
                print("Use nomes únicos e contatos visíveis na lateral. Não navegue manualmente durante a operação.")
                tasks = [asyncio.create_task(scanner(page, targets, incoming, lock, store)),
                         asyncio.create_task(worker(incoming, outgoing, client, store)),
                         asyncio.create_task(dispatcher(page, outgoing, lock, store, enabled, allowed))]
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await context.close()
    finally:
        await client.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Bot encerrado; respostas pendentes não serão reenviadas no reinício.")
