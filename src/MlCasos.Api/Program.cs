using Microsoft.Extensions.ML;
using MlCasos.Modelos;

var builder = WebApplication.CreateBuilder(args);

// Padrão: models/tickets.zip na raiz do repositório
var caminho = Path.GetFullPath(
    builder.Configuration["Modelos:Tickets"]
    ?? Path.Combine(builder.Environment.ContentRootPath,
        "..", "..", "models", "tickets.zip"));

builder.Services
    .AddPredictionEnginePool<TicketEntrada, TicketSaida>()
    .FromFile(
        modelName: "tickets",
        filePath: caminho,
        watchForChanges: true);
builder.Services.AddSingleton<DetectorPicos>();

var app = builder.Build();

app.MapPost("/tickets/classificar", (
    TextoRequest req,
    PredictionEnginePool<TicketEntrada, TicketSaida> pool) =>
{
    if (string.IsNullOrWhiteSpace(req.Texto))
        return Results.BadRequest("Informe o texto do ticket.");

    var r = pool.Predict("tickets",
        new TicketEntrada { Texto = req.Texto });

    var confianca = r.Score.Max();
    return Results.Ok(new
    {
        categoria = r.Categoria,
        confianca = Math.Round(confianca, 2),
        triagem = confianca < 0.6f
    });
});

app.MapPost("/vendas/anomalias", (
    VendasRequest req, DetectorPicos detector) =>
{
    if (req.Valores is null || req.Valores.Length < 8)
        return Results.BadRequest("Envie ao menos 8 valores.");

    return Results.Ok(detector.Detectar(req.Valores));
});

app.Run();

public record TextoRequest(string? Texto);
public record VendasRequest(float[]? Valores);
