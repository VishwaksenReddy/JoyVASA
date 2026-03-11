from torch import nn


class MotionExtractorExportWrapper(nn.Module):
    output_names = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return tuple(outputs[name] for name in self.output_names)


class WarpingModuleExportWrapper(nn.Module):
    output_names = ["occlusion_map", "deformation", "out"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feature_3d, kp_driving, kp_source):
        outputs = self.model(feature_3d, kp_driving=kp_driving, kp_source=kp_source)
        return outputs["occlusion_map"], outputs["deformation"], outputs["out"]


class StitchingHeadExportWrapper(nn.Module):
    output_names = ["delta"]

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feat):
        return self.head(feat)


class MotionAudioFeatureExportWrapper(nn.Module):
    output_names = ["audio_feat"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        return self.model.extract_audio_feature(audio)


class MotionDenoiserExportWrapper(nn.Module):
    output_names = ["motion_feat_target"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator):
        step = step.long()
        return self.model.denoising_net(
            motion_feat,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            step,
            indicator,
        )
