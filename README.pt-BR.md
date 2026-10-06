[English](README.md) | Português

# mlnet-casos-dotnet

> **Início rápido**

```bash
dotnet build
dotnet run --project src/MlCasos.Treino --no-build
dotnet run --project src/MlCasos.Api --no-build
```

Precisa apenas do SDK do .NET 10. Sem nuvem, sem GPU. Detalhes em [Como executar](#como-executar).

Dois casos práticos e pequenos de [ML.NET](https://learn.microsoft.com/dotnet/machine-learning/) em .NET 10 (`Microsoft.ML` 5.0.0), expostos por uma Minimal API do ASP.NET Core:

1. **Classificação de tickets de suporte** (classificação de texto multiclasse): modelo treinado a partir de `data/tickets.tsv` (96 tickets rotulados em português: `Acesso`, `Cobranca`, `Entrega`, `Tecnico`), salvo em `models/tickets.zip` e servido com `PredictionEnginePool` (`Microsoft.Extensions.ML`).
2. **Detecção de picos de vendas** (detecção de anomalias em série temporal): `DetectIidSpike` do `Microsoft.ML.TimeSeries`, aplicado à série enviada na requisição.

## Pré-requisitos

- [SDK do .NET 10](https://dotnet.microsoft.com/download)

## Como executar

Na raiz do repositório:

```bash
dotnet build
dotnet run --project src/MlCasos.Treino --no-build
dotnet run --project src/MlCasos.Api --no-build
```

O treino imprime a acurácia no conjunto de teste (cerca de 78% neste dataset minúsculo; varia um pouco conforme a versão) e grava `models/tickets.zip`. A API lê esse arquivo (configurável em `Modelos:Tickets`) e o recarrega quando ele muda.

Chame os endpoints (a porta aparece na inicialização; `5088` abaixo):

```bash
curl -X POST http://localhost:5088/tickets/classificar \
  -H "Content-Type: application/json" \
  -d '{"texto":"Quero a segunda via do boleto"}'

curl -X POST http://localhost:5088/vendas/anomalias \
  -H "Content-Type: application/json" \
  -d '{"valores":[100,102,98,101,99,103,100,97,240,101,99,102,100,98,20,101]}'
```

Saída esperada:

```
{"categoria":"Cobranca","confianca":0.99,"triagem":false}
[{"indice":8,"valor":240,"pValor":1E-08},{"indice":14,"valor":20,"pValor":1E-08}]
```

## Estrutura

```
mlnet-casos-dotnet.slnx
data/tickets.tsv          tickets rotulados (separado por tab)
src/
  MlCasos.Modelos/        tipos compartilhados, DetectorPicos
  MlCasos.Treino/         console: avalia e salva o modelo
  MlCasos.Api/            Minimal API (PredictionEnginePool)
```

Identificadores, categorias e mensagens ficam em português de propósito. O dataset é pequeno e existe só para o exemplo rodar offline; um classificador real precisa de muito mais dados.

## Licença

[MIT](LICENSE). Autor: Carlos Eduardo Seidl.
