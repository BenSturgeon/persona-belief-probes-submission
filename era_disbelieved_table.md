Llama 3.3 70B, Layer 30, mean over 15 historical personas. Both `era_true` and `era_disbelieved` columns are absolute probe scores. **Reading the table:** under every induction method, both columns drop together by roughly the same amount, so the `shift (eT−eD)` gap stays small and no row is significant (p > 0.14). Induction suppresses true-today statements broadly but does not differentially penalise the era-rejected ones.

| Condition      | era_true | era_disbelieved | shift (eT−eD) |  t(14) |    p |
|----------------|---------:|----------------:|--------------:|-------:|-----:|
| k=0 (neutral)  |   +0.40  |          +0.16  |        +0.24  | +1.53  | 0.15 |
| ICL k=10       |   −1.76  |          −1.72  |        −0.04  | −0.40  | 0.70 |
| ICL k=32       |   −1.43  |          −1.33  |        −0.10  | −1.19  | 0.25 |
| system prompt  |   −1.21  |          −0.99  |        −0.21  | −1.56  | 0.14 |
| SFT            |   −0.67  |          −0.59  |        −0.08  | −0.70  | 0.50 |
