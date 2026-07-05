using System.Reflection;
using System.Runtime.Loader;

if (args.Length == 0)
{
    Console.Error.WriteLine(
        "Usage: Sts2ApiProbe <path-to-sts2.dll> [full-type-name ...]"
    );
    return 2;
}

var assemblyPath = Path.GetFullPath(args[0]);
var assemblyDirectory = Path.GetDirectoryName(assemblyPath)
    ?? throw new InvalidOperationException("Assembly directory is missing.");
var loadContext = new AssemblyLoadContext("sts2-api-probe");
loadContext.Resolving += (context, name) =>
{
    var dependencyPath = Path.Combine(
        assemblyDirectory,
        $"{name.Name}.dll"
    );
    return File.Exists(dependencyPath)
        ? context.LoadFromAssemblyPath(dependencyPath)
        : null;
};
var assembly = loadContext.LoadFromAssemblyPath(assemblyPath);
var typeNames = args.Skip(1).ToArray();
if (typeNames.Length >= 2 && typeNames[0] == "--find")
{
    foreach (var type in assembly.GetTypes()
        .Where(type => typeNames.Skip(1).Any(pattern =>
            type.FullName?.Contains(
                pattern,
                StringComparison.OrdinalIgnoreCase
            ) is true
        ))
        .OrderBy(type => type.FullName))
    {
        Console.WriteLine(type.FullName);
    }
    return 0;
}
if (typeNames.Length == 0)
{
    typeNames = [
        "MegaCrit.Sts2.Core.Entities.Players.Player",
        "MegaCrit.Sts2.Core.Runs.IRunState",
        "MegaCrit.Sts2.Core.Entities.Cards.CardPile",
        "MegaCrit.Sts2.Core.Models.CardModel",
        "MegaCrit.Sts2.Core.Rewards.CardReward",
        "MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen",
    ];
}

const BindingFlags flags = BindingFlags.Public
    | BindingFlags.NonPublic
    | BindingFlags.Instance
    | BindingFlags.Static
    | BindingFlags.DeclaredOnly;

foreach (var typeName in typeNames)
{
    var type = assembly.GetType(typeName);
    Console.WriteLine($"\n## {typeName}");
    if (type is null)
    {
        Console.WriteLine("TYPE NOT FOUND");
        continue;
    }
    foreach (var property in type.GetProperties(flags)
        .OrderBy(property => property.Name))
    {
        Console.WriteLine(
            $"PROPERTY {Visibility(property.GetMethod ?? property.SetMethod)} "
            + $"{property.PropertyType.FullName} {property.Name}"
        );
    }
    foreach (var field in type.GetFields(flags).OrderBy(field => field.Name))
    {
        Console.WriteLine(
            $"FIELD {FieldVisibility(field)} {field.FieldType.FullName} "
            + field.Name
        );
    }
    foreach (var method in type.GetMethods(flags)
        .Where(method => !method.IsSpecialName)
        .OrderBy(method => method.Name))
    {
        var parameters = string.Join(
            ", ",
            method.GetParameters().Select(parameter =>
                $"{parameter.ParameterType.FullName} {parameter.Name}"
            )
        );
        Console.WriteLine(
            $"METHOD {Visibility(method)} {method.ReturnType.FullName} "
            + $"{method.Name}({parameters})"
        );
    }
}

return 0;

static string Visibility(MethodBase? member)
{
    if (member is null)
    {
        return "unknown";
    }
    if (member.IsPublic)
    {
        return "public";
    }
    if (member.IsFamily || member.IsFamilyOrAssembly)
    {
        return "protected";
    }
    if (member.IsAssembly)
    {
        return "internal";
    }
    return "private";
}

static string FieldVisibility(FieldInfo field)
{
    if (field.IsPublic)
    {
        return "public";
    }
    if (field.IsFamily || field.IsFamilyOrAssembly)
    {
        return "protected";
    }
    if (field.IsAssembly)
    {
        return "internal";
    }
    return "private";
}
