# Notas de teste do rag (Rust)

Este arquivo existe para validar busca semântica em português.

O índice usa embeddings locais (LM Studio, Ollama ou OpenAI-compatível). A
indexação é manual por padrão: você roda `rag index` quando quiser, e apenas
arquivos alterados são reprocessados (barato).

As estatísticas mostram custo (tokens embarcados, tempo) versus utilidade
(buscas servidas, contexto entregue versus leitura completa de arquivos,
e a taxa de utilidade julgada pelo jev).

Arquivos P0 (código-fonte e markdown) são processados primeiro e vencem P1
na busca (score multiplicado por 0.9).
