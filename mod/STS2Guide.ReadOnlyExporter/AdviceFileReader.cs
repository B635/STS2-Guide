using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

internal sealed record AdviceFileResult<T>(T? Value, AdviceAvailability Availability) where T : class;

/// <summary>Read a single atomic advice snapshot. No previous value survives rejection.</summary>
internal static class AdviceFileReader
{
    internal static AdviceFileResult<T> Read<T>(string path, Func<JsonElement, T?> validate) where T : class
    {
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var value = validate(document.RootElement);
            return value is null
                ? new(null, AdviceAvailability.Unavailable)
                : new(value, AdviceAvailabilityText.FromValidatedRecommendation(
                    document.RootElement.GetProperty("recommendation")));
        }
        catch (Exception exception) when (exception is FileNotFoundException or DirectoryNotFoundException)
        {
            return new(null, AdviceAvailability.Waiting);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException
            or JsonException or InvalidOperationException or KeyNotFoundException or FormatException)
        {
            return new(null, AdviceAvailability.Unavailable);
        }
    }
}
