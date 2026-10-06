using Microsoft.ML;
using MlCasos.Modelos;

var ml = new MLContext(seed: 1);

var dados = ml.Data.LoadFromTextFile<TicketEntrada>(
    "data/tickets.tsv", hasHeader: true);

var pipeline = ml.Transforms.Conversion
    .MapValueToKey("Label", nameof(TicketEntrada.Categoria))
    .Append(ml.Transforms.Text.FeaturizeText(
        "Features", nameof(TicketEntrada.Texto)))
    .Append(ml.MulticlassClassification.Trainers
        .SdcaMaximumEntropy())
    .Append(ml.Transforms.Conversion
        .MapKeyToValue("PredictedLabel"));

// 1) mede: treina com 75% e avalia nos 25% restantes
var divisao = ml.Data.TrainTestSplit(
    dados, testFraction: 0.25, seed: 1);
var teste = pipeline.Fit(divisao.TrainSet)
    .Transform(divisao.TestSet);
var m = ml.MulticlassClassification.Evaluate(teste);
Console.WriteLine($"Acurácia (teste): {m.MicroAccuracy:P0}");

// 2) publica: treina com tudo e salva o modelo em .zip
var modelo = pipeline.Fit(dados);
Directory.CreateDirectory("models");
ml.Model.Save(modelo, dados.Schema, "models/tickets.zip");
Console.WriteLine("Modelo salvo em models/tickets.zip");
