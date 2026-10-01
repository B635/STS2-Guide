// Read-only STS2 assembly fact extractor (development tool, never shipped).
// Decodes metadata and IL only: it never loads the assembly for execution,
// never instantiates game models and never calls game methods.
using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

var dumpClass = args.Length == 3 && args[1] == "--dump" ? args[2] : null;
if (args.Length != 2 && dumpClass is null)
{
    Console.Error.WriteLine(
        "Usage: Sts2FactExtractor <path-to-sts2.dll> <output.json>\n"
        + "       Sts2FactExtractor <path-to-sts2.dll> --dump <ClassName>");
    return 2;
}

const string ExtractorVersion = "sts2-fact-extractor-v4";
const string CardTagEnum = "MegaCrit.Sts2.Core.Entities.Cards.CardTag";
const string DynamicVarsNamespace = "MegaCrit.Sts2.Core.Localization.DynamicVars.";
const string CardTypeGetter = "MegaCrit.Sts2.Core.Models.CardModel::get_Type";
const string CardTypeEnum = "MegaCrit.Sts2.Core.Entities.Cards.CardType";
var targets = new Dictionary<string, string>
{
    ["MegaCrit.Sts2.Core.Models.Relics"] = "relic",
    ["MegaCrit.Sts2.Core.Models.Cards"] = "card",
    ["MegaCrit.Sts2.Core.Models.Powers"] = "power",
};

var assemblyPath = Path.GetFullPath(args[0]);
var assemblyBytes = File.ReadAllBytes(assemblyPath);
var assemblySha = Convert.ToHexString(SHA256.HashData(assemblyBytes)).ToLowerInvariant();

using var pe = new PEReader(ImmutableArray.Create(assemblyBytes));
var reader = pe.GetMetadataReader();
var names = new Names(reader);
var opcodes = typeof(OpCodes)
    .GetFields(BindingFlags.Static | BindingFlags.Public)
    .Select(field => (OpCode)field.GetValue(null)!)
    .ToDictionary(op => (ushort)op.Value);
var cardTypeNames = EnumNames(reader, names, CardTypeEnum);
var cardTagNames = EnumNames(reader, names, CardTagEnum);

var results = new List<(string kind, string name, object entry)>();
foreach (var handle in reader.TypeDefinitions)
{
    var definition = reader.GetTypeDefinition(handle);
    if (!definition.GetDeclaringType().IsNil)
    {
        continue;
    }
    var ns = reader.GetString(definition.Namespace);
    if (!targets.TryGetValue(ns, out var kind))
    {
        continue;
    }

    var tree = new List<TypeDefinitionHandle>();
    CollectTree(reader, handle, tree);
    tree.Sort((a, b) => string.CompareOrdinal(names.Type(a), names.Type(b)));

    // Override name -> metadata token(s) of the overriding method(s).
    var overrides = new SortedDictionary<string, string>(StringComparer.Ordinal);
    foreach (var methodHandle in definition.GetMethods())
    {
        var method = reader.GetMethodDefinition(methodHandle);
        var attributes = method.Attributes;
        if ((attributes & MethodAttributes.Virtual) != 0
            && (attributes & MethodAttributes.NewSlot) == 0)
        {
            var name = reader.GetString(method.Name);
            var token = MetadataTokens.GetToken(methodHandle).ToString("X8");
            overrides[name] = overrides.TryGetValue(name, out var existing)
                ? existing + "," + token
                : token;
        }
    }

    var calls = new SortedSet<string>(StringComparer.Ordinal);
    var fields = new SortedSet<string>(StringComparer.Ordinal);
    var typeRefs = new SortedSet<string>(StringComparer.Ordinal);
    var strings = new SortedSet<string>(StringComparer.Ordinal);
    var cardTypeChecks = new SortedSet<string>(StringComparer.Ordinal);
    var canonicalVars = new List<object>();
    var upgradeDeltas = new List<object>();
    var canonicalTags = new SortedSet<string>(StringComparer.Ordinal);
    int[]? cardCtorArgs = null;
    var canonical = new StringBuilder();

    foreach (var typeHandle in tree)
    {
        var typeName = names.Type(typeHandle);
        canonical.Append("type ").Append(typeName).Append('\n');
        var methods = reader.GetTypeDefinition(typeHandle).GetMethods()
            .Select(h => (handle: h, key: names.MethodKey(h)))
            .OrderBy(entry => entry.key, StringComparer.Ordinal);
        foreach (var (methodHandle, key) in methods)
        {
            canonical.Append("method ").Append(key).Append('\n');
            var method = reader.GetMethodDefinition(methodHandle);
            if (method.RelativeVirtualAddress == 0)
            {
                continue;
            }
            var il = pe.GetMethodBody(method.RelativeVirtualAddress).GetILBytes()!;
            var instructions = Decode(il, opcodes, names, reader);
            if (typeHandle.Equals(handle))
            {
                var methodName = reader.GetString(method.Name);
                if (methodName == "get_CanonicalVars")
                {
                    canonicalVars.AddRange(DecodeVars(instructions));
                }
                else if (methodName == "OnUpgrade")
                {
                    upgradeDeltas.AddRange(DecodeUpgrades(instructions));
                }
                else if (methodName == "get_CanonicalTags")
                {
                    for (var i = 1; i < instructions.Count; i++)
                    {
                        if (instructions[i].operand.EndsWith("::Add", StringComparison.Ordinal)
                            && LoadedInt(instructions[i - 1]) is int tag)
                        {
                            canonicalTags.Add(cardTagNames.TryGetValue(tag, out var name) ? name : tag.ToString());
                        }
                    }
                }
                else if (methodName == ".ctor" && kind == "card")
                {
                    // CardModel(cost, type, rarity, target, ...): the int
                    // arguments loaded straight before the base constructor.
                    var baseCall = instructions.FindIndex(ins => ins.operand == "MegaCrit.Sts2.Core.Models.CardModel::.ctor");
                    if (baseCall > 0)
                    {
                        var ints = new List<int>();
                        for (var i = baseCall - 1; i >= 0 && LoadedInt(instructions[i]) is int value; i--)
                        {
                            ints.Insert(0, value);
                        }
                        cardCtorArgs = ints.ToArray();
                    }
                }
            }
            for (var index = 0; index < instructions.Count; index++)
            {
                var (op, operand, operandKind) = instructions[index];
                canonical.Append(op.Name).Append(' ').Append(operand).Append('\n');
                switch (operandKind)
                {
                    case "method":
                        calls.Add(operand);
                        if (operand == CardTypeGetter)
                        {
                            for (var next = index + 1;
                                 next < Math.Min(index + 4, instructions.Count);
                                 next++)
                            {
                                var value = LoadedInt(instructions[next]);
                                if (value is int constant)
                                {
                                    cardTypeChecks.Add(cardTypeNames.TryGetValue(constant, out var label)
                                        ? label
                                        : constant.ToString());
                                    break;
                                }
                            }
                        }
                        break;
                    case "field":
                        fields.Add(operand);
                        break;
                    case "type":
                        typeRefs.Add(operand);
                        break;
                    case "string":
                        strings.Add(operand);
                        break;
                }
            }
        }
    }

    if (dumpClass is not null)
    {
        if (reader.GetString(definition.Name) == dumpClass)
        {
            Console.Write(canonical.ToString());
            return 0;
        }
        continue;
    }

    var ownPrefix = names.Type(handle);
    // Keep only game-semantic references; async plumbing, BCL collections,
    // hover tips and visual nodes carry no mechanic facts. The fingerprint
    // above still covers every instruction.
    static bool Semantic(string reference) =>
        reference.StartsWith("MegaCrit.Sts2.Core.", StringComparison.Ordinal)
        && !reference.StartsWith("MegaCrit.Sts2.Core.HoverTips.", StringComparison.Ordinal)
        && !reference.StartsWith("MegaCrit.Sts2.Core.Nodes.", StringComparison.Ordinal)
        && !reference.StartsWith("MegaCrit.Sts2.Core.Audio.", StringComparison.Ordinal);
    results.Add((kind, reader.GetString(definition.Name), new
    {
        @class = reader.GetString(definition.Name),
        kind,
        @namespace = ns,
        @base = definition.BaseType.IsNil ? null : names.Entity(definition.BaseType),
        @abstract = (definition.Attributes & TypeAttributes.Abstract) != 0,
        overrides = overrides.Keys.ToArray(),
        override_tokens = overrides,
        calls = calls.Where(c => !c.StartsWith(ownPrefix, StringComparison.Ordinal) && Semantic(c)).ToArray(),
        fields = fields.Where(f => !f.StartsWith(ownPrefix, StringComparison.Ordinal) && Semantic(f)).ToArray(),
        types = typeRefs.Where(Semantic).ToArray(),
        strings = strings.ToArray(),
        card_type_checks = cardTypeChecks.ToArray(),
        canonical_vars = canonicalVars,
        canonical_tags = canonicalTags.ToArray(),
        card_ctor = cardCtorArgs is { Length: >= 2 }
            ? new
            {
                cost = cardCtorArgs[0],
                type = cardTypeNames.TryGetValue(cardCtorArgs[1], out var ctorTypeName) ? ctorTypeName : cardCtorArgs[1].ToString(),
                args = cardCtorArgs,
            }
            : null,
        upgrade_deltas = upgradeDeltas,
        fingerprint = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString()))
        ).ToLowerInvariant(),
    }));
}

if (dumpClass is not null)
{
    Console.Error.WriteLine($"class {dumpClass} not found");
    return 1;
}

var payload = new
{
    extractor_version = ExtractorVersion,
    assembly_sha256 = assemblySha,
    types = results
        .OrderBy(result => result.kind, StringComparer.Ordinal)
        .ThenBy(result => result.name, StringComparer.Ordinal)
        .Select(result => result.entry)
        .ToArray(),
};
var output = JsonSerializer.Serialize(payload, new JsonSerializerOptions
{
    WriteIndented = true,
    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
});
File.WriteAllText(args[1], output.Replace("\r\n", "\n") + "\n", new UTF8Encoding(false));
Console.WriteLine($"{results.Count} types, assembly {assemblySha}");
return 0;

// Best-effort constant tracking over straight-line IL: decimal literals are
// built from int loads (``new decimal(int)`` or the five-argument form) and
// then passed to a DynamicVar constructor or ``UpgradeValueBy``. Anything the
// pattern cannot read stays absent and must be reviewed from the IL dump.
static decimal? DecimalFrom(List<long> ints, string ctor)
{
    if (ctor.EndsWith("/1", StringComparison.Ordinal) && ints.Count >= 1)
    {
        return ints[^1];
    }
    if (ctor.EndsWith("/5", StringComparison.Ordinal) && ints.Count >= 5)
    {
        var lo = (int)ints[^5];
        var mid = (int)ints[^4];
        var hi = (int)ints[^3];
        var negative = ints[^2] != 0;
        var scale = (byte)ints[^1];
        return new decimal(lo, mid, hi, negative, scale);
    }
    return null;
}

static IEnumerable<(string op, string operand, decimal? value)> DecimalStream(
    List<(OpCode op, string operand, string kind)> instructions)
{
    var ints = new List<long>();
    foreach (var instruction in instructions)
    {
        var loaded = LoadedInt(instruction);
        if (loaded is int value)
        {
            ints.Add(value);
            yield return ("int", "", value);
            continue;
        }
        if (instruction.op.Name == "newobj"
            && instruction.operand.StartsWith("System.Decimal::.ctor", StringComparison.Ordinal))
        {
            yield return ("decimal", "", DecimalFrom(ints, instruction.operand));
            ints.Clear();
            continue;
        }
        if (instruction.op.Name == "ldsfld" && instruction.operand.StartsWith("System.Decimal::", StringComparison.Ordinal))
        {
            var constant = instruction.operand["System.Decimal::".Length..] switch
            {
                "Zero" => 0m,
                "One" => 1m,
                "MinusOne" => -1m,
                _ => (decimal?)null,
            };
            yield return ("decimal", "", constant);
            ints.Clear();
            continue;
        }
        yield return (instruction.op.Name ?? "", instruction.operand, null);
        if (instruction.op.Name != "ldstr")
        {
            ints.Clear();
        }
    }
}

static List<object> DecodeVars(List<(OpCode op, string operand, string kind)> instructions)
{
    var result = new List<object>();
    decimal? value = null;
    decimal? lastInt = null;
    string? name = null;
    foreach (var (op, operand, constant) in DecimalStream(instructions))
    {
        if (op == "int")
        {
            lastInt = constant;
        }
        else if (op == "decimal")
        {
            value = constant;
        }
        else if (op == "ldstr")
        {
            name = operand;
        }
        else if (op == "newobj" && operand.StartsWith(DynamicVarsNamespace, StringComparison.Ordinal))
        {
            result.Add(new
            {
                type = operand[DynamicVarsNamespace.Length..].Replace("::.ctor", ""),
                name,
                // Some vars take an int directly (e.g. CardsVar(1)).
                value = value ?? lastInt,
            });
            value = null;
            lastInt = null;
            name = null;
        }
    }
    return result;
}

static List<object> DecodeUpgrades(List<(OpCode op, string operand, string kind)> instructions)
{
    var result = new List<object>();
    string? variable = null;
    string? name = null;
    decimal? value = null;
    foreach (var (op, operand, constant) in DecimalStream(instructions))
    {
        if (op == "int")
        {
            continue;
        }
        if (op == "decimal")
        {
            value = constant;
        }
        else if (op == "ldstr")
        {
            name = operand;
        }
        else if (operand.Contains("DynamicVarSet::get_", StringComparison.Ordinal))
        {
            var getter = operand[(operand.IndexOf("::get_", StringComparison.Ordinal) + 6)..];
            variable = getter == "Item" ? name : getter;
        }
        else if (operand.EndsWith("::UpgradeValueBy", StringComparison.Ordinal))
        {
            result.Add(new { var = variable, delta = value });
            variable = null;
            value = null;
        }
        else if ((op == "call" || op == "callvirt")
            && operand.StartsWith("MegaCrit.Sts2.Core.", StringComparison.Ordinal)
            && !operand.Contains("::get_", StringComparison.Ordinal))
        {
            // Any other game call in OnUpgrade (cost change, keyword add...)
            // is listed so a reviewer sees every upgrade effect.
            result.Add(new { call = operand, value });
            value = null;
        }
    }
    return result;
}

static void CollectTree(MetadataReader reader, TypeDefinitionHandle handle, List<TypeDefinitionHandle> tree)
{
    tree.Add(handle);
    foreach (var nested in reader.GetTypeDefinition(handle).GetNestedTypes())
    {
        CollectTree(reader, nested, tree);
    }
}

static Dictionary<int, string> EnumNames(MetadataReader reader, Names names, string enumFullName)
{
    var result = new Dictionary<int, string>();
    var enumName = enumFullName[(enumFullName.LastIndexOf('.') + 1)..];
    var found = false;
    foreach (var handle in reader.TypeDefinitions)
    {
        var type = reader.GetTypeDefinition(handle);
        if (names.Type(handle) != enumFullName)
        {
            continue;
        }
        found = true;
        foreach (var fieldHandle in type.GetFields())
        {
            var field = reader.GetFieldDefinition(fieldHandle);
            var constantHandle = field.GetDefaultValue();
            if (constantHandle.IsNil)
            {
                continue;
            }
            var blob = reader.GetBlobReader(reader.GetConstant(constantHandle).Value);
            var value = blob.Length switch
            {
                1 => blob.ReadByte(),
                2 => blob.ReadInt16(),
                _ => blob.ReadInt32(),
            };
            result[value] = $"{enumName}.{reader.GetString(field.Name)}";
        }
    }
    if (!found)
    {
        throw new InvalidOperationException($"enum {enumFullName} not found");
    }
    return result;
}

static int? LoadedInt((OpCode op, string operand, string kind) instruction)
{
    var name = instruction.op.Name ?? "";
    if (name == "ldc.i4.m1")
    {
        return -1;
    }
    if (name.StartsWith("ldc.i4.", StringComparison.Ordinal)
        && int.TryParse(name["ldc.i4.".Length..], out var inline))
    {
        return inline;
    }
    if ((name == "ldc.i4.s" || name == "ldc.i4") && int.TryParse(instruction.operand, out var value))
    {
        return value;
    }
    return null;
}

static List<(OpCode op, string operand, string kind)> Decode(
    byte[] il,
    Dictionary<ushort, OpCode> opcodes,
    Names names,
    MetadataReader reader)
{
    var result = new List<(OpCode, string, string)>();
    var offset = 0;
    while (offset < il.Length)
    {
        ushort code = il[offset++];
        if (code == 0xfe)
        {
            code = (ushort)(0xfe00 | il[offset++]);
        }
        var op = opcodes[code];
        var operand = "";
        var kind = "";
        switch (op.OperandType)
        {
            case OperandType.InlineNone:
                break;
            case OperandType.ShortInlineI:
            case OperandType.ShortInlineBrTarget:
            case OperandType.ShortInlineVar:
                operand = ((sbyte)il[offset]).ToString();
                offset += 1;
                break;
            case OperandType.InlineVar:
                operand = BitConverter.ToUInt16(il, offset).ToString();
                offset += 2;
                break;
            case OperandType.InlineI:
            case OperandType.InlineBrTarget:
                operand = BitConverter.ToInt32(il, offset).ToString();
                offset += 4;
                break;
            case OperandType.InlineI8:
                operand = BitConverter.ToInt64(il, offset).ToString();
                offset += 8;
                break;
            case OperandType.ShortInlineR:
                operand = BitConverter.ToSingle(il, offset).ToString("R");
                offset += 4;
                break;
            case OperandType.InlineR:
                operand = BitConverter.ToDouble(il, offset).ToString("R");
                offset += 8;
                break;
            case OperandType.InlineSwitch:
                var count = BitConverter.ToInt32(il, offset);
                var targets = Enumerable.Range(0, count)
                    .Select(i => BitConverter.ToInt32(il, offset + 4 + 4 * i));
                operand = string.Join(",", targets);
                offset += 4 + 4 * count;
                break;
            case OperandType.InlineString:
                operand = reader.GetUserString(
                    MetadataTokens.UserStringHandle(BitConverter.ToInt32(il, offset) & 0xffffff));
                kind = "string";
                offset += 4;
                break;
            case OperandType.InlineMethod:
                operand = names.Entity(MetadataTokens.EntityHandle(BitConverter.ToInt32(il, offset)));
                kind = "method";
                offset += 4;
                break;
            case OperandType.InlineField:
                operand = names.Entity(MetadataTokens.EntityHandle(BitConverter.ToInt32(il, offset)));
                kind = "field";
                offset += 4;
                break;
            case OperandType.InlineType:
            case OperandType.InlineTok:
                operand = names.Entity(MetadataTokens.EntityHandle(BitConverter.ToInt32(il, offset)));
                kind = "type";
                offset += 4;
                break;
            default:
                operand = names.Entity(MetadataTokens.EntityHandle(BitConverter.ToInt32(il, offset)));
                offset += 4;
                break;
        }
        result.Add((op, operand, kind));
    }
    return result;
}

sealed class Names : ISignatureTypeProvider<string, object?>
{
    private readonly MetadataReader _reader;

    public Names(MetadataReader reader) => _reader = reader;

    public string Type(TypeDefinitionHandle handle)
    {
        var type = _reader.GetTypeDefinition(handle);
        var parent = type.GetDeclaringType();
        var prefix = parent.IsNil ? _reader.GetString(type.Namespace) : Type(parent);
        return prefix + "." + _reader.GetString(type.Name);
    }

    public string MethodKey(MethodDefinitionHandle handle)
    {
        var method = _reader.GetMethodDefinition(handle);
        var signature = method.DecodeSignature(this, null);
        return _reader.GetString(method.Name)
            + "(" + string.Join(",", signature.ParameterTypes) + ")" + signature.ReturnType;
    }

    public string Entity(EntityHandle handle)
    {
        switch (handle.Kind)
        {
            case HandleKind.TypeDefinition:
                return Type((TypeDefinitionHandle)handle);
            case HandleKind.TypeReference:
                return GetTypeFromReference(_reader, (TypeReferenceHandle)handle, 0);
            case HandleKind.TypeSpecification:
                return _reader.GetTypeSpecification((TypeSpecificationHandle)handle).DecodeSignature(this, null);
            case HandleKind.MethodDefinition:
                var method = _reader.GetMethodDefinition((MethodDefinitionHandle)handle);
                return Type(method.GetDeclaringType()) + "::" + _reader.GetString(method.Name);
            case HandleKind.FieldDefinition:
                var field = _reader.GetFieldDefinition((FieldDefinitionHandle)handle);
                return Type(field.GetDeclaringType()) + "::" + _reader.GetString(field.Name);
            case HandleKind.MemberReference:
                var member = _reader.GetMemberReference((MemberReferenceHandle)handle);
                var owner = Entity(member.Parent);
                var memberName = _reader.GetString(member.Name);
                if (owner == "System.Decimal" && memberName == ".ctor")
                {
                    // Keep the overload: new decimal(int) vs the 5-part form.
                    var arity = member.DecodeMethodSignature(this, null).ParameterTypes.Length;
                    return $"{owner}::.ctor/{arity}";
                }
                return owner + "::" + memberName;
            case HandleKind.MethodSpecification:
                var specification = _reader.GetMethodSpecification((MethodSpecificationHandle)handle);
                var arguments = specification.DecodeSignature(this, null);
                return Entity(specification.Method) + "<" + string.Join(",", arguments) + ">";
            default:
                return handle.Kind.ToString();
        }
    }

    public string GetArrayType(string elementType, ArrayShape shape) => elementType + "[" + new string(',', shape.Rank - 1) + "]";
    public string GetByReferenceType(string elementType) => elementType + "&";
    public string GetFunctionPointerType(MethodSignature<string> signature) => "fnptr";
    public string GetGenericInstantiation(string genericType, ImmutableArray<string> typeArguments) => genericType + "<" + string.Join(",", typeArguments) + ">";
    public string GetGenericMethodParameter(object? genericContext, int index) => "!!" + index;
    public string GetGenericTypeParameter(object? genericContext, int index) => "!" + index;
    public string GetModifiedType(string modifier, string unmodifiedType, bool isRequired) => unmodifiedType;
    public string GetPinnedType(string elementType) => elementType;
    public string GetPointerType(string elementType) => elementType + "*";
    public string GetPrimitiveType(PrimitiveTypeCode typeCode) => typeCode.ToString();
    public string GetSZArrayType(string elementType) => elementType + "[]";
    public string GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind) => Type(handle);
    public string GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind)
    {
        var reference = reader.GetTypeReference(handle);
        var scope = reference.ResolutionScope;
        var prefix = scope.Kind == HandleKind.TypeReference
            ? GetTypeFromReference(reader, (TypeReferenceHandle)scope, 0)
            : reader.GetString(reference.Namespace);
        return prefix + "." + reader.GetString(reference.Name);
    }
    public string GetTypeFromSpecification(MetadataReader reader, object? genericContext, TypeSpecificationHandle handle, byte rawTypeKind)
        => reader.GetTypeSpecification(handle).DecodeSignature(this, genericContext);
}
