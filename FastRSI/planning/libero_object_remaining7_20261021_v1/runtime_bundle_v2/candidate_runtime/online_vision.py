from online_vision_frozen import Vision as FrozenVision
from fastrsi.visual_recovery.perception import vision_class
from pathlib import Path
Vision = vision_class(FrozenVision, Path(__file__).with_name("STATIC_SOURCE_POSE.json"),
                      Path(__file__).with_name("INITIAL_LOCATOR_REGISTRY.json"))
