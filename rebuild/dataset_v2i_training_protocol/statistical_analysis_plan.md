# Plano estatístico prospectivo — Dataset-v2I

## Unidade e métricas

A unidade de inferência é a unidade de atribuição/componente congelada em v2G. O teste contém 73 objetos em 61 unidades. Cada objeto mantém previsão e probabilidades; objetos da mesma unidade não são réplicas independentes. Métricas pontuais sobre objetos são descritivas; intervalos e testes amostram unidades completas.

Relatar, para cada seed, arquitetura e representação, balanced accuracy, macro-F1 e accuracy, além de precisão, recall, F1 e suporte por classe e matriz de confusão. Relatar todos os seeds e média ± desvio-padrão amostral (ddof=1), sem escolher melhor seed. Balanced accuracy é a média de recall das classes com suporte verdadeiro positivo; macro-F1 inclui as quatro classes, usando zero quando o denominador é zero. Reportar degenerescências e suporte zero.

## Bootstrap pareado

Usar NumPy PCG64, 20.000 replicatas, seed 20261021, intervalos percentis bilaterais de 95% com quantil linear. Cada replicata sorteia 61 unidades com reposição; todos os objetos da unidade permanecem juntos. Usar exatamente a mesma sequência de amostras em modelos, representações, métricas e seeds. Calcular diferenças pareadas por seed e a média das três diferenças. A incerteza é condicional aos checkpoints congelados e não incorpora integralmente variação de treino.

## Randomização e multiplicidade

Usar 20.000 replicatas de randomização pareada no nível da unidade: trocar conjuntamente as previsões dos dois modelos para todos os objetos da unidade e todos os três seeds. P bilateral `(1 + # replicatas tão ou mais extremas)/(20.001)`. Usar as mesmas trocas em métricas e comparações relacionadas.

Holm step-down, alfa 0,05, em famílias separadas: (i) três pares de arquiteturas × três métricas em A (9 hipóteses); (ii) três pares × três métricas em B, A_outer e B_outer (27); (iii) cada controle B/A_outer/B_outer contra A × três arquiteturas × três métricas (27 por controle). Intervalos percentis são marginais, não simultâneos após Holm. Não usar teste t com n=3 seeds.

Os dois modelos metadata/FOV fixos são sempre relatados, com comparações pareadas por unidade contra cada arquitetura em A, apenas como controles descritivos; não participam da seleção.

## Isolamento e selagem

Workers de predição recebem imagens e membership, nunca truth. Selar e hashear todas as previsões antes de disponibilizar truth ao processo de scoring. O bootstrap e a randomização só ocorrem depois da selagem. Avaliar o teste uma única vez; qualquer resultado não pode motivar treino, substituição, seleção ou alteração.

## Componentes com múltiplos objetos

Preservar previsões por objeto e todas as probabilidades. Não colapsar por majority vote, subdividir unidades, nem tratar 73 objetos como 73 unidades independentes. Métricas estratificadas por FOV/extensão reportam suporte e bins vazios.
