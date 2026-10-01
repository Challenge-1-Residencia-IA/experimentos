import pandas as pd

# No pandas 3 com pyarrow instalado, as strings usam o motor de regex do Arrow (RE2), em que `\w` e `\b`
# só reconhecem ASCII: "terça-feira" deixa de casar com `\w+-feira`. Guardar as strings como objetos
# Python mantém a semântica do módulo `re`, da qual as regex deste pacote e dos notebooks dependem.
pd.set_option("mode.string_storage", "python")
