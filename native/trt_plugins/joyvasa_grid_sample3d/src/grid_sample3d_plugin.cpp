#include "grid_sample3d_kernel.hpp"

#include <NvInferPluginBase.h>
#include <NvInferRuntime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace nvinfer1;

constexpr AsciiChar kPluginName[] = "JoyVASAGridSample3D";
constexpr AsciiChar kPluginVersion[] = "1";
constexpr AsciiChar kPluginNamespace[] = "com.joyvasa";
constexpr AsciiChar kModeFieldName[] = "mode";
constexpr AsciiChar kPaddingModeFieldName[] = "padding_mode";
constexpr AsciiChar kAlignCornersFieldName[] = "align_corners";

#if defined(_WIN32)
#define JOYVASA_TRT_PLUGIN_EXPORT __declspec(dllexport)
#else
#define JOYVASA_TRT_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

int32_t parseIntField(PluginFieldCollection const* fc, AsciiChar const* name, int32_t defaultValue) noexcept
{
    if (fc == nullptr || fc->fields == nullptr)
    {
        return defaultValue;
    }

    for (int32_t index = 0; index < fc->nbFields; ++index)
    {
        PluginField const& field = fc->fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0 && field.data != nullptr)
        {
            return *static_cast<int32_t const*>(field.data);
        }
    }
    return defaultValue;
}

bool isSupportedDataType(DataType type) noexcept
{
    return type == DataType::kFLOAT || type == DataType::kHALF;
}

bool validateDims(Dims const& feature, Dims const& grid, Dims const& output) noexcept
{
    return feature.nbDims == 5 && grid.nbDims == 5 && output.nbDims == 5 && grid.d[4] == 3 && output.d[0] == feature.d[0]
        && output.d[1] == feature.d[1] && output.d[2] == grid.d[1] && output.d[3] == grid.d[2] && output.d[4] == grid.d[3]
        && grid.d[0] == feature.d[0];
}

class JoyVASAGridSample3DPlugin final
    : public IPluginV3
    , public IPluginV3OneCore
    , public IPluginV3OneBuild
    , public IPluginV3OneRuntime
{
public:
    JoyVASAGridSample3DPlugin(int32_t mode, int32_t paddingMode, int32_t alignCorners) noexcept
        : mMode(mode)
        , mPaddingMode(paddingMode)
        , mAlignCorners(alignCorners)
    {
        refreshMetadata();
        refreshSerializedFields();
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
        }
    }

    IPluginV3* clone() noexcept override
    {
        return new JoyVASAGridSample3DPlugin(mMode, mPaddingMode, mAlignCorners);
    }

    AsciiChar const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override
    {
        return kPluginNamespace;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs,
        int32_t nbInputs,
        DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        DimsExprs* outputs,
        int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        (void) shapeInputs;
        (void) nbShapeInputs;
        (void) exprBuilder;
        if (nbInputs != 2 || nbOutputs != 1 || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        outputs[0].nbDims = 5;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = inputs[1].d[1];
        outputs[0].d[3] = inputs[1].d[2];
        outputs[0].d[4] = inputs[1].d[3];
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos,
        DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= (nbInputs + nbOutputs))
        {
            return false;
        }

        DynamicPluginTensorDesc const& desc = inOut[pos];
        if (desc.desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        if (pos == 0)
        {
            return isSupportedDataType(desc.desc.type);
        }

        if (pos == 1)
        {
            return desc.desc.type == inOut[0].desc.type || desc.desc.type == DataType::kFLOAT;
        }

        return desc.desc.type == inOut[0].desc.type;
    }

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        if (mMode != 0 || mPaddingMode != 0 || mAlignCorners != 0)
        {
            return 1;
        }
        if (in == nullptr || out == nullptr || nbInputs != 2 || nbOutputs != 1)
        {
            return 1;
        }
        if (!validateDims(in[0].desc.dims, in[1].desc.dims, out[0].desc.dims))
        {
            return 1;
        }
        if (!isSupportedDataType(in[0].desc.type) || !isSupportedDataType(out[0].desc.type))
        {
            return 1;
        }
        return 0;
    }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        (void) inputs;
        (void) nbInputs;
        (void) outputs;
        (void) nbOutputs;
        return 0;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes,
        int32_t nbOutputs,
        DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 2)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    AsciiChar const* getMetadataString() noexcept override
    {
        return mMetadata.c_str();
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        (void) tactics;
        (void) nbTactics;
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        (void) context;
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        refreshSerializedFields();
        return &mSerializedFieldCollection;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* inputs,
        int32_t nbInputs,
        PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 2 || nbOutputs != 1)
        {
            return 1;
        }
        return validateDims(inputs[0].dims, inputs[1].dims, outputs[0].dims) ? 0 : 1;
    }

    int32_t enqueue(
        PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override
    {
        (void) workspace;
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        return enqueueGridSample3D(inputDesc[0], inputDesc[1], outputDesc[0], inputs[0], inputs[1], outputs[0], stream);
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        (void) tactic;
        return 0;
    }

private:
    void refreshMetadata()
    {
        mMetadata = "mode=" + std::to_string(mMode) + ",padding_mode=" + std::to_string(mPaddingMode)
            + ",align_corners=" + std::to_string(mAlignCorners);
    }

    void refreshSerializedFields()
    {
        mSerializedFields = {
            PluginField{kModeFieldName, &mMode, PluginFieldType::kINT32, 1},
            PluginField{kPaddingModeFieldName, &mPaddingMode, PluginFieldType::kINT32, 1},
            PluginField{kAlignCornersFieldName, &mAlignCorners, PluginFieldType::kINT32, 1},
        };
        mSerializedFieldCollection.nbFields = static_cast<int32_t>(mSerializedFields.size());
        mSerializedFieldCollection.fields = mSerializedFields.data();
    }

    int32_t mMode{0};
    int32_t mPaddingMode{0};
    int32_t mAlignCorners{0};
    std::string mMetadata;
    std::vector<PluginField> mSerializedFields;
    PluginFieldCollection mSerializedFieldCollection{};
};

class JoyVASAGridSample3DPluginCreator final : public IPluginCreatorV3One
{
public:
    JoyVASAGridSample3DPluginCreator()
    {
        mFieldNames = {
            PluginField{kModeFieldName, nullptr, PluginFieldType::kINT32, 1},
            PluginField{kPaddingModeFieldName, nullptr, PluginFieldType::kINT32, 1},
            PluginField{kAlignCornersFieldName, nullptr, PluginFieldType::kINT32, 1},
        };
        mFieldCollection.nbFields = static_cast<int32_t>(mFieldNames.size());
        mFieldCollection.fields = mFieldNames.data();
    }

    AsciiChar const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override
    {
        return kPluginNamespace;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFieldCollection;
    }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        (void) name;
        (void) phase;
        int32_t mode = parseIntField(fc, kModeFieldName, 0);
        int32_t paddingMode = parseIntField(fc, kPaddingModeFieldName, 0);
        int32_t alignCorners = parseIntField(fc, kAlignCornersFieldName, 0);
        if (mode != 0 || paddingMode != 0 || alignCorners != 0)
        {
            return nullptr;
        }
        return new JoyVASAGridSample3DPlugin(mode, paddingMode, alignCorners);
    }

private:
    std::vector<PluginField> mFieldNames;
    PluginFieldCollection mFieldCollection{};
};

JoyVASAGridSample3DPluginCreator gPluginCreator;

} // namespace

extern "C" {

JOYVASA_TRT_PLUGIN_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder* finder) noexcept
{
    (void) finder;
}

JOYVASA_TRT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) noexcept
{
    static nvinfer1::IPluginCreatorInterface* creators[] = {&gPluginCreator};
    nbCreators = 1;
    return creators;
}

} // extern "C"
