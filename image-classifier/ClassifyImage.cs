using System.Net;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ImageSharpImage = SixLabors.ImageSharp.Image;

namespace image_classifier;

public class ClassifyImage {
    private static readonly string ModelPath =
        Path.Combine(AppContext.BaseDirectory, "model", "mobilenetv2.onnx");

    private static readonly string LabelsPath =
        Path.Combine(AppContext.BaseDirectory, "model", "labels.txt");

    private static readonly Lazy<InferenceSession> Session = new(() => new InferenceSession(ModelPath));
    private static readonly Lazy<string[]> Labels = new(() => File.ReadAllLines(LabelsPath));

    // ImageNet normalization
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    [Function("ClassifyImage")]
    public async Task<HttpResponseData> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post")] HttpRequestData req) {
        var bytes = await ReadBody(req);
        if (bytes.Length == 0) {
            var bad = req.CreateResponse(HttpStatusCode.BadRequest);
            await bad.WriteAsJsonAsync(new { error = "Send raw image bytes in the POST body (binary)." });
            return bad;
        }

        using Image<Rgb24> img = ImageSharpImage.Load<Rgb24>(bytes);
        img.Mutate(x => x.Resize(224, 224));

        var inputName = Session.Value.InputMetadata.Keys.First();
        var outputName = Session.Value.OutputMetadata.Keys.First();

        var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        FillNchwNormalized(img, input);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, input)
        };

        using var results = Session.Value.Run(inputs);
        var logits = results.First(r => r.Name == outputName).AsEnumerable<float>().ToArray();

        var probs = Softmax(logits);
        int idx = ArgMax(probs);

        var ok = req.CreateResponse(HttpStatusCode.OK);
        await ok.WriteAsJsonAsync(new {
            label = idx < Labels.Value.Length ? Labels.Value[idx] : $"class_{idx}",
            confidence = probs[idx],
            index = idx
        });
        return ok;
    }

    private static async Task<byte[]> ReadBody(HttpRequestData req) {
        using var ms = new MemoryStream();
        await req.Body.CopyToAsync(ms);
        return ms.ToArray();
    }

    private static void FillNchwNormalized(Image<Rgb24> img, DenseTensor<float> t) {
        img.ProcessPixelRows(accessor => {
            for (int y = 0; y < 224; y++) {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < 224; x++) {
                    var p = row[x];

                    float r = (p.R / 255f - Mean[0]) / Std[0];
                    float g = (p.G / 255f - Mean[1]) / Std[1];
                    float b = (p.B / 255f - Mean[2]) / Std[2];

                    t[0, 0, y, x] = r;
                    t[0, 1, y, x] = g;
                    t[0, 2, y, x] = b;
                }
            }
        });
    }

    private static int ArgMax(float[] v) {
        int idx = 0;
        float best = v[0];
        for (int i = 1; i < v.Length; i++) {
            if (v[i] > best) { best = v[i]; idx = i; }
        }
        return idx;
    }

    private static float[] Softmax(float[] logits) {
        float max = logits.Max();
        var exps = new float[logits.Length];
        double sum = 0;

        for (int i = 0; i < logits.Length; i++) {
            double e = Math.Exp(logits[i] - max);
            exps[i] = (float)e;
            sum += e;
        }

        for (int i = 0; i < exps.Length; i++)
            exps[i] = (float)(exps[i] / sum);

        return exps;
    }
}
