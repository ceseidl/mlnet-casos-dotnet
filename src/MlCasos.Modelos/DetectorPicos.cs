using Microsoft.ML;
using Microsoft.ML.Data;

namespace MlCasos.Modelos;

public class PontoVenda
{
    public float Valor { get; set; }
}

public class PontoAlerta
{
    // [0] alerta (0 ou 1), [1] score, [2] p-value
    [VectorType(3)]
    public double[] Alerta { get; set; } = [];
}

public record Anomalia(int Indice, float Valor, double PValor);

public class DetectorPicos
{
    private readonly MLContext _ml = new(seed: 1);

    public IReadOnlyList<Anomalia> Detectar(
        float[] valores, double confianca = 95)
    {
        var dados = _ml.Data.LoadFromEnumerable(
            valores.Select(v => new PontoVenda { Valor = v }));

        var pipeline = _ml.Transforms.DetectIidSpike(
            outputColumnName: nameof(PontoAlerta.Alerta),
            inputColumnName: nameof(PontoVenda.Valor),
            confidence: confianca,
            pvalueHistoryLength: Math.Max(5, valores.Length / 4));

        var saida = pipeline.Fit(dados).Transform(dados);
        var alertas = _ml.Data
            .CreateEnumerable<PontoAlerta>(
                saida, reuseRowObject: false)
            .ToList();

        return alertas
            .Select((a, i) => (a, i))
            .Where(x => x.a.Alerta[0] == 1)
            .Select(x => new Anomalia(
                x.i, valores[x.i], x.a.Alerta[2]))
            .ToList();
    }
}
