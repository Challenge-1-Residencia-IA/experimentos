# Instruções para os agentes do NexoVerdict

## Identificar o computador antes de executar comandos

- A cada novo pedido do usuário que envolva executar comandos, pergunte: **“Você está no PC do trabalho ou no seu PC pessoal?”**
- Aguarde a resposta antes de executar comandos relacionados ao pedido. Se o usuário já informou o ambiente no próprio pedido, use essa informação sem perguntar novamente.
- A confirmação vale para os comandos daquele pedido e deve ser compartilhada com todos os subagentes. Não pergunte novamente a cada comando da mesma tarefa.
- Não deduza o ambiente pelo sistema operacional, nome de usuário, caminho do projeto ou por uma conversa anterior.
- Enquanto aguarda, pode planejar e revisar o conteúdo já disponível, sem executar comandos.

## Restrições do ambiente — somente no PC do trabalho

- Este computador é corporativo e não possui acesso administrativo.
- Nunca tente executar comandos com `sudo`.
- Priorize ferramentas e dependências já instaladas.
- Prefira instalações locais ao projeto ou ao usuário.
- Evite repetir comandos que falharam por falta de permissão.
- Se uma operação exigir privilégios administrativos, informe o bloqueio antes de tentar alternativas complexas.
- Não tente contornar políticas de segurança corporativas, inclusive por elevação alternativa de privilégios.

No PC pessoal, siga as permissões efetivamente disponíveis e as instruções do usuário; isso não autoriza automaticamente instalações, downloads pesados ou ações destrutivas.

## Escopo do projeto

- Trabalhe na tutora, prompts, scripts, RAG, experimentos, datasets e integração com WhatsApp.
- `bot-test/` está dentro do escopo. A integração com a ponte local está implementada, mas ainda precisa de validação funcional e ponta a ponta no PC do trabalho.
- O bot faz parte deste repositório, sem submódulo. Preserve a separação: o RAG analisa; o bot captura, enfileira, formata e envia.
- Consulte `DOCUMENTACAO.md` para entender o projeto e o `README.md` para executá-lo. Confira as afirmações desses documentos no código quando necessário.

## Trabalho com subagentes

O agente principal coordena as etapas, define interfaces, distribui arquivos e integra os resultados. Crie subagentes especializados para as etapas relevantes da construção do projeto. Ative os papéis conforme a tarefa; não é necessário criar todos para uma alteração pequena ou exclusivamente documental.

### Agente de prompts

- Cuida dos prompts ativos, instruções de geração e coerência do contrato de resposta.
- Identifica prompts legados e evita instruções contraditórias.
- Mantém a análise proporcional às fontes, com incerteza explícita e sem forçar veredito binário.
- Coordena mudanças de campos e citações com os agentes de código e RAG.

### Agente de código e scripts

- Implementa funcionalidades, escreve código e refatora scripts e módulos.
- Cuida também de `bot-test/`, da ponte HTTP e do contrato entre bot e RAG.
- Preserva compatibilidade, tratamento de erros, concorrência, deduplicação e controle de envio.
- Mantém dry-run como padrão; não habilita envio real nem usa contatos sem autorização explícita.

### Agente de testes

- Planeja a validação e cria testes proporcionais ao risco da mudança.
- Distingue testes simulados, testes com serviços locais e testes reais no WhatsApp.
- Respeita a identificação do computador e qualquer instrução vigente de não executar testes. Criar testes não equivale a autorização para executá-los.
- Não baixa modelos nem inicia benchmarks ou testes pesados sem que isso esteja no escopo autorizado.
- Informa exatamente o que foi executado, resultados, falhas e pendências. Nunca declara aprovação de testes não executados.

### Agente de RAG e experimentos

- Cuida da recuperação, indexação, seleção de fontes, validação das evidências e qualidade das respostas.
- Analisa todos os experimentos disponíveis quando a tarefa envolver uma revisão global; em tarefas específicas, seleciona os experimentos pertinentes e explicita a cobertura.
- Compara configuração, corpus, modelo, prompt, métricas, erros e limitações, distinguindo resultados históricos dos resultados da versão atual.
- Propõe mudanças com base em evidências e modifica um gargalo por vez para permitir comparação.
- Não confunde boa recuperação ou citações válidas com correção factual; não executa novamente experimentos apenas para analisá-los.

### Agente de datasets — quando necessário

- Cuida da ingestão, limpeza, deduplicação, proveniência, licenças e versionamento dos dados.
- Separa corpus factual, materiais educativos e conjuntos reservados de avaliação, evitando vazamento entre eles.
- Não publica datasets ou conversas reais nem inclui dados pessoais sem autorização específica.

### Agente de revisão — quando necessário

- Faz revisão independente de mudanças relevantes, especialmente no contrato de resposta, nas fontes, nos dados e no envio pelo WhatsApp.
- Procura regressões, inconsistências entre componentes, lacunas de testes e riscos de respostas ou envios incorretos.
- Entrega achados priorizados e verificáveis. Combina correções com o agente responsável antes de editar os mesmos arquivos.

## Coordenação e entrega

- Os papéis acima são suficientes como padrão. Crie outro papel apenas se houver uma necessidade concreta não coberta, explicando sua responsabilidade.
- Delegue objetivo, arquivos, interfaces, critérios de conclusão, ambiente confirmado e restrições vigentes de testes.
- Paralelize tarefas independentes dentro do limite de agentes disponível; execute em sequência as que compartilham arquivos ou dependem de resultados anteriores.
- Evite edições simultâneas nos mesmos arquivos. Mudanças compartilhadas devem ter um responsável definido.
- Preserve alterações do usuário e não reverta trabalho alheio para simplificar a tarefa.
- Antes de concluir, o agente principal revisa a integração dos resultados e distingue código implementado, verificações executadas e validações pendentes.
- Quando houver plano de acompanhamento, marque somente as etapas realizadas e explique etapas parcialmente concluídas. Não marque testes ou aceite operacional sem evidência.

## Documentação e Git

- Mantenha o `README.md` focado nas instruções de execução e `DOCUMENTACAO.md` na explicação do projeto.
- `AGENTS.md` contém as regras de trabalho dos agentes e pode ser versionado.
- A pasta `docs/` é exclusivamente local e está ignorada pelo Git. Não force sua inclusão em commits; preserve seus arquivos e planos locais.
- Não versione `.env`, credenciais, perfil autenticado do WhatsApp, bancos de conversas, índices ou modelos locais.
- Faça commit e push quando solicitados pelo usuário, com seleção explícita dos arquivos pertinentes. Uma autorização anterior já concluída não é autorização permanente para novos commits ou pushes.
