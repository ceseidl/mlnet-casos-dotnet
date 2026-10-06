English | [Português](README.pt-BR.md)

# mlnet-casos-dotnet

> **Quick start**

```bash
dotnet build
dotnet run --project src/MlCasos.Treino --no-build
dotnet run --project src/MlCasos.Api --no-build
```

Needs only the .NET 10 SDK. No cloud, no GPU. Details in [How to run](#how-to-run).

Two small, practical [ML.NET](https://learn.microsoft.com/dotnet/machine-learning/) cases in .NET 10 (`Microsoft.ML` 5.0.0), served by an ASP.NET Core Minimal API:

1. **Support ticket classification** (multiclass text classification): a model trained from `data/tickets.tsv` (96 labelled tickets in Portuguese: `Acesso`, `Cobranca`, `Entrega`, `Tecnico`), saved as `models/tickets.zip` and served through `PredictionEnginePool` (`Microsoft.Extensions.ML`).
2. **Sales spike detection** (anomaly detection in a time series): `DetectIidSpike` from `Microsoft.ML.TimeSeries`, applied to the series sent in the request.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## How to run

From the repository root:

```bash
dotnet build
dotnet run --project src/MlCasos.Treino --no-build
dotnet run --project src/MlCasos.Api --no-build
```

The trainer prints the hold-out accuracy (about 78% on this tiny dataset; it varies slightly by version) and writes `models/tickets.zip`. The API reads that file (override with the `Modelos:Tickets` setting) and reloads it when it changes.

Call the endpoints (the port is printed at startup; `5088` below):

```bash
curl -X POST http://localhost:5088/tickets/classificar \
  -H "Content-Type: application/json" \
  -d '{"texto":"Quero a segunda via do boleto"}'

curl -X POST http://localhost:5088/vendas/anomalias \
  -H "Content-Type: application/json" \
  -d '{"valores":[100,102,98,101,99,103,100,97,240,101,99,102,100,98,20,101]}'
```

Expected output:

```
{"categoria":"Cobranca","confianca":0.99,"triagem":false}
[{"indice":8,"valor":240,"pValor":1E-08},{"indice":14,"valor":20,"pValor":1E-08}]
```

## Structure

```
mlnet-casos-dotnet.slnx
data/tickets.tsv          labelled tickets (tab separated)
src/
  MlCasos.Modelos/        shared types, DetectorPicos
  MlCasos.Treino/         console app: evaluate and save the model
  MlCasos.Api/            Minimal API (PredictionEnginePool)
```

Identifiers, categories and messages are in Portuguese on purpose. The dataset is small and exists only to make the example run offline; a real classifier needs far more data.

## License

[MIT](LICENSE). Author: Carlos Eduardo Seidl.
