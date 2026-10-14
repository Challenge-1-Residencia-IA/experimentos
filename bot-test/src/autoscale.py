"""Compatibilidade com o bot importado: o RAG local opera com um worker fixo."""


async def autoscale_workers(*args, **kwargs):
    return  # Desativado para preservar ordem e não disputar o Ollama.
