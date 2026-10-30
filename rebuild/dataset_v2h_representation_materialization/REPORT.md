# Dataset-v2H — materialização determinística de representações

## Resultado

A materialização terminou para os 370 objetos elegíveis do split v2G. A é a representação científica primária; B é o controle obrigatório de campo comum. Foram também criados os controles estáticos A_outer e B_outer. Nenhum tensor usa produto de reserva, rejeitado, diferido ou bloqueado.

O digest vinculante do split é `1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf`; o inventário v2G verificado é `1819213f7e863390cb25b9869f0a187c835da87fe543d1eee9ad33d4b9db600d`. Os protocolos v2F foram verificados pelos digests canônicos. Todos os 1.110 FITS congelados foram verificados por tamanho e SHA-256 antes da decodificação.

## Contagens congeladas

| Split | Objetos | FITS master | unidades de atribuição |
|---|---:|---:|---:|
| train | 222 | 666 | 222 |
| validation | 75 | 225 | 75 |
| test | 73 | 219 | 61 |

Contagens por classe e split constam em `sealed_label_manifest.json` e `leakage_validation.json`. Cada classe supera os pisos 21/6/10. Papéis e IDs de grupo não aparecem nos nomes dos tensores; rótulos ficam em manifesto separado.

## Representações e controles

Foram gravados 1.480 tensores float32 contíguos CHW, shape 3×224×224, intervalo [-1,1], canais blue/red/NIR: 370 por variante A, B, A_outer e B_outer. Cada arquivo tem tamanho e SHA-256 em seu manifesto e no índice combinado. Total: 891,315,200 bytes. Raiz Merkle SHA-256: `111411afe4a4f20bafca15f2d5f10be2e1321d065cce2dc4b30bc4121f73d001`. Digest canônico vinculado do conjunto: `089048e74a683100b1a317be6f32a5924e203129264ba9441bb0320275ad6021`.

B aplica o campo TAN comum de 4 arcmin, amostragem efetiva 16×16, suavização gaussiana angular congelada, quadratura 8×8, interpolação bilinear explícita e replicação 14×. O suporte foi verificado em cada ponto amostrado. A_outer e B_outer usam mascaramento e percentis apenas da região não mascarada conforme v2F. Os controles de metadados/FOV e de estratificação por FOV/extent foram registrados para uso futuro; nenhum modelo de controle foi ajustado.

D4 foi validado separadamente com arrays sintéticos e travamento de canais; nenhum aumento aleatório foi aplicado aos tensores armazenados. Validação/teste usam identidade. Uma reconstrução integral refez os 1.480 fluxos `.npy` em memória: zero diferenças de tamanho ou SHA-256.

## Integridade e limites

Validação independente confirmou zero cruzamento de objetos físicos, unidades de atribuição, conteúdo exato, derivados ou componentes de sobreposição. Os 17 componentes permanecem indivisíveis. Todos os arquivos v2A–v2G foram comparados com o snapshot anterior à decodificação e permanecem byte a byte idênticos.

Esta fase só materializou e validou representações. Não houve aumento aleatório, treinamento, inferência, carregamento de checkpoints, pontuação, rede ou ajuste de controles. Os controles de shortcut continuam obrigatórios, qualquer que seja o resultado futuro.

**Estado:** representações materializadas e determinísticas, prontas para revisão e para um protocolo de treinamento separado, ainda não congelado.
