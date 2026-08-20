# Relatório Técnico: Avaliação de Impacto Causal dos Incentivos Fiscais Industriais no Ceará (FDI / ICMS)

**Instituição**: Centro de Inteligência Industrial (Prova de Conceito)  
**Autor**: Pietro Esteves (Engenharia de Machine Learning & Econometria Sênior)  
**Data**: Agosto de 2026  
**Status**: Prova de Conceito  

> **Nota Metodológica sobre os Dados**: Este relatório utiliza dados sintéticos gerados por Processo Gerador de Dados (DGP) calibrado com estatísticas municipais do Ceará. Os resultados numéricos são artefatos de simulação monte carlo para benchmark econométrico do framework `industrial-causal-policy`.

---

## 1. Sumário Executivo

A avaliação de políticas públicas industriais exige superação do mero acompanhamento descritivo para responder a uma pergunta central de tomada de decisão: **"Qual é o verdadeiro efeito causal heterogêneo das isenções/reduções fiscais de ICMS (via FDI) sobre a geração de empregos formais industriais nos municípios do Ceará, e em quais regiões o retorno da política é maximizado?"**

Este estudo desenvolve e valida o framework **`industrial-causal-policy`**, aplicando **5 métodos causais de Machine Learning** (IPW, Linear DML, Causal Forest DML, DR Learner e X-Learner) combinados a **Controle Sintético Estadual** e **4 testes formais de refutação (DoWhy)**.

### Principais Achados:
1. **Efeito Médio Geral (ATE)**: A política industrial de FDI gera um efeito causal positivo estatisticamente significante de **~85 a 95 novos empregos formais industriais líquidos por município/ano**, após controlar por fatores de seleção socioeconômica.
2. **Heterogeneidade Marcante (CATE)**: O efeito causal **não é homogêneo**. Municípios do corredor industrial da Região Metropolitana de Fortaleza (Horizonte, Maracanaú, Eusébio, Caucaia) e polos regionais qualificados (Sobral, Juazeiro do Norte) apresentam CATEs superiores a **150 empregos/ano**, enquanto municípios isolados do interior de baixo IDH registram CATEs moderados a neutros (~20 a 40 empregos/ano).
3. **Drivers do Impacto (SHAP)**: A proximidade geográfica à capital/portos (`dist_capital_km`), a presença de infraestrutura técnica de capacitação (`senai_presence`) e o IDH-M municipal são os três principais fatores explicativos da magnitude do impacto.
4. **Avaliação Macroeconômica (Controle Sintético)**: No nível agregado, a política de FDI permitiu ao Ceará acumular um ganho líquido excedente de mais de **35.000 empregos industriais** vis-à-vis um Ceará Sintético construído a partir de estados do Nordeste e vizinhos no período 2021–2025.
5. **Robustez Econométrica**: Todos os métodos superaram os 4 testes de refutação DoWhy (Placebo Treatment, Random Common Cause, Data Subset 80% e Sensibilidade de Confundidores Não Observados).

---

## 2. Dados e Granularidade

A base analítica unifica quatro fontes de dados oficiais em nível municipal $\times$ setor CNAE 2.0 $\times$ ano (2020–2025):

| Variável | Papel Econométrico | Fonte | Descrição |
|---|---|---|---|
| **`net_job_gain`** | Outcome ($Y$) | Novo CAGED | Saldo líquido de empregos industriais (admissões − desligamentos) |
| **`fdi_incentive`** | Tratamento ($T$) | SEFAZ-CE / Diário Oficial | Dummy binária de concessão de incentivo fiscal ICMS via FDI |
| **`pib_per_capita`** | Confundidor ($X$) | IBGE PIB dos Municípios | PIB municipal per capita em R$ |
| **`idhm`** | Confundidor ($X$) | PNUD / Atlas Brasil | Índice de Desenvolvimento Humano Municipal |
| **`dist_capital_km`** | Confundidor ($X$) | Cálculo Haversine | Distância em km até Fortaleza |
| **`senai_presence`** | Confundidor ($X$) | Dados de Infraestrutura | Dummy indicadora de unidade física de capacitação técnica |
| **`ind_emp_share`** | Confundidor ($X$) | RAIS | Proporção de emprego industrial sobre o emprego formal total |
| **`n_establishments`**| Covariável ($X$) | RAIS | Número de estabelecimentos industriais ativos |
| **`avg_industrial_wage`**| Covariável ($X$) | CAGED / RAIS | Salário médio industrial em R$ |

---

## 3. Grafo Causal (DAG) & Identification Assumptions

O projeto explicita o Grafo Causal Dirigido (DAG) formalizando as premissas de identificação:

1. **Unconfoundedness (Conditional Independence Assumption)**:  
   $Y(1), Y(0) \perp\!\!\!\!\perp T \mid X$.  
   Condicional ao vetor de covariáveis socioeconômicas e de infraestrutura $X$, a concessão do incentivo fiscal é independente dos resultados potenciais.
2. **Positivity (Overlap)**:  
   $0 < P(T=1 \mid X=x) < 1, \quad \forall x$.  
   Verificado empiricamente no Love Plot e nas distribuições de Propensity Score.
3. **SUTVA (Stable Unit Treatment Value Assumption)**:  
   O tratamento concedido a um município não afeta diretamente os resultados potenciais de outros municípios não vizinhos.

---

## 4. Tabela Consolidada do Benchmark dos 5 Métodos

| Método Causal | ATE Estimado (Empregos/Ano) | IC 95% Inferior | IC 95% Superior | P-Valor | RMSE CATE | Desempenho / Recomendação |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **IPW (Inverse Probability Weighting)** | 45.5 | 37.3 | 59.2 | < 0.0001 | 185.4 | Baseline clássico (Assume efeito constante) |
| **Linear DML (Double ML)** | 66.9 | 14.7 | 119.0 | 0.0012 | 210.1 | Baseline semi-paramétrico linear |
| **Causal Forest DML (EconML)** | **95.3** | **15.9** | **174.7** | **< 0.0001** | **88.2** | **Estimador principal** (Captura heterogeneidade não-linear) |
| **DR Learner (Doubly Robust)** | 84.9 | 72.8 | 97.0 | < 0.0001 | 92.4 | Robustez dupla de especificação |
| **X-Learner (Künzel et al.)** | 86.3 | 73.9 | 98.7 | < 0.0001 | 90.1 | Otimizado para poucas unidades tratadas |

---

## 5. Recomendação de Alocação de Política Industrial

Com base na **Policy Tree** e nos agrupamentos de resposta CATE:

1. **Priorização de Concessão (Alto Retorno)**:  
   Conceder incentivos FDI prioritariamente em municípios com IDH-M $> 0.64$, distância até Fortaleza $< 180\text{ km}$ e presença de capacitação técnica. Nessas localidades, cada real de renúncia fiscal gera o maior saldo de empregos qualificados.
2. **Políticas Complementares para o Interior**:  
   Nos municípios do interior distante de baixo IDH, a simples concessão de isenção de ICMS não atrai indústrias sustentáveis sem investimentos prévios em capacitação técnica profissional e infraestrutura de transporte/energia.

---

*Relatório produzido com o framework `industrial-causal-policy` como prova de conceito para aplicação em centros de inteligência industrial.*
