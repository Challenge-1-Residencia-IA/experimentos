# Executar o bot

Siga o [README principal](../README.md) para instalar dependências, preparar
o RAG, iniciar a ponte e configurar o contato em `bot-test/.env`.

Na raiz de `experimentos`, com `.venv` ativada e a ponte pronta:

```powershell
python bot-test/botfinal.py
```

Mantenha `BOT_ENABLE_SEND=false` para revisar as prévias. A descrição do fluxo,
do contrato e das limitações está em [DOCUMENTACAO.md](../DOCUMENTACAO.md).
