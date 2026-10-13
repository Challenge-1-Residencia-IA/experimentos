# Bot WhatsApp Web com tutor RAG local

O scanner e o despachante Playwright usam a ponte HTTP do NexoVerdict na mesma máquina. A análise vem do Qwen pelo Ollama, com recuperação de fontes pelo RAG. Mensagens são processadas uma por vez. O envio real fica desligado por padrão.

## Preparação

1. Prepare o repositório `experimentos` seguindo a seção "Ponte local para o bot" no README dele. Configure Ollama, Qwen, corpus e índice Chroma ou Neon antes de iniciar o bot.
2. Neste repositório, crie um ambiente Python 3.10+ e instale `pip install -r requirements.txt` e `playwright install chromium`.
3. Copie `.env.example` para `.env`. Preencha `BOT_TARGET_CHAT` com o nome exato do contato de teste. O padrão `WHATSAPP_SEND_ENABLED=false` apenas mostra as respostas no console.
4. Inicie a ponte no repositório `experimentos`: `python3 -m scripts.rag_bridge`. Confirme `http://127.0.0.1:8765/health` com `status: ready`.
5. Inicie o bot com `python3 botfinal.py`. O navegador abre o WhatsApp Web; autentique a sessão. O scanner tenta abrir somente o contato alvo se ele estiver visível na lista lateral. Se não aparecer `Contato alvo aberto e conferido`, abra esse contato manualmente. Espere `Chat alvo observado; histórico visível ignorado` antes de pedir uma nova mensagem à outra conta.

O bot não inicia se a ponte estiver indisponível. A ponte é apenas o canal local entre os dois processos. Interrompa o bot antes da ponte para não deixar análises pendentes. O perfil do navegador, `.env` e o SQLite local devem ficar fora do Git. O banco guarda identificadores derivados por hash e marcadores de conteúdo, sem persistir o texto integral das conversas.

O modo padrão `BOT_SCAN_MODE=single_chat` procura apenas a linha lateral cujo nome ou número corresponde a `BOT_TARGET_CHAT` e confere o cabeçalho após abri-la. Não percorre outras conversas para leitura. Mensagens já visíveis ao abrir o chat são ignoradas; peça uma mensagem **nova** de outra conta após a linha `Chat alvo observado`. O modo anterior de varrer conversas pode ser selecionado com `BOT_SCAN_MODE=legacy`, mas apresentou falhas de associação entre chats nos testes e não deve ser usado para envio real.

## Envio controlado

Para envio real, defina **ambas** as variáveis no `.env`:

```text
WHATSAPP_SEND_ENABLED=true
WHATSAPP_ALLOWED_CONTACTS=Nome Exato do Contato
```

Use somente contatos previamente autorizados. Os demais permanecem em simulação. Confirme primeiro as respostas em `dry-run` e faça o teste real controlado no PC do trabalho. O bot confere o nome do chat imediatamente antes do envio, reserva a tentativa no SQLite, procura a mensagem enviada no DOM e só então a marca como confirmada. Uma tentativa sem confirmação fica `pendente` e exige conferência manual antes de qualquer novo envio; não há reenvio automático.

Para localizar tentativas pendentes, abra o banco local `data/bot.db` e consulte `SELECT m.chat_nome, e.timestamp FROM envios e JOIN mensagens m ON m.mensagem_id=e.mensagem_id WHERE e.status='pendente';`. Confira o chat no WhatsApp antes de qualquer intervenção manual.

`RAG_BRIDGE_URL` deve apontar para `http://127.0.0.1:8765` na mesma máquina. `RAG_BRIDGE_TIMEOUT` (padrão 200 s) deve ser maior que o prazo da ponte: `RAG_BRIDGE_TIMEOUT_SECONDS` (padrão 90 s) por alegação separada, até duas. `RAG_QUEUE_MAX` limita a fila (padrão 10). Áudio e mídia recebem aviso de não suporte; mensagens textuais e links passam pelo RAG. Falhas do RAG produzem apenas um aviso seguro, sem usar Groq.

A resposta para WhatsApp põe URLs HTTP(S) completas em linhas próprias para o aplicativo reconhecê-las como links. Mostra todos os links disponíveis no resultado da análise, sem repetir URLs: fontes citadas, páginas consultadas ou descartadas, checagens candidatas e resultados de busca ainda não verificados. Os rótulos deixam claro quais links sustentam a resposta e quais servem apenas para consulta. A contagem de resultados da busca pode incluir entradas sem URL utilizável; essas não viram link.

A ponte bloqueia mensagens com e-mail, telefone, CPF, CNPJ ou parâmetros sensíveis de URL antes de pesquisar na web. O bot pede que a pessoa remova esses dados e reenvie a mensagem. A detecção automática cobre padrões comuns, mas não todos os dados pessoais; revise o uso em conversas reais.

## Validação rápida e retorno

Execute apenas os testes direcionados às alterações: `python3 -m unittest test_database.py` e os casos de `test_rag_client.py`. Não abra o WhatsApp para testes unitários. Para voltar ao comportamento seguro, defina `WHATSAPP_SEND_ENABLED=false` e reinicie o bot. Para voltar à versão de código anterior, pare os dois processos e use o commit anterior do bot; não reutilize sessões com respostas pendentes.

O teste ponta a ponta com Ollama, índice, ponte e WhatsApp ainda precisa ocorrer no PC do trabalho antes de considerar a integração pronta. Confirme casos favorável, contrário, inconclusivo e link em `dry-run`, depois um envio a contato autorizado, sem duplicatas.
