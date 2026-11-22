"""Text prompts for zero-shot CLIP scoring."""
from __future__ import annotations


def human(key: str) -> str:
    """EPIC class key -> plain words: 'board:chopping' -> 'chopping board', 'turn-on' -> 'turn on'."""
    parts = key.replace("_", " ").split(":")
    if len(parts) > 1:
        parts = parts[1:] + parts[:1]
    return " ".join(parts).replace("-", " ").strip()


# One or more prompts per video-level label (FPVSum categories + cooking).
LABEL_PROMPTS = {
    "bike": ["a first-person video of riding a bicycle on a trail", "a POV view from the handlebars of a mountain bike"],
    "bikepolo": ["a first-person video of bike polo", "people playing polo on bicycles with mallets"],
    "boxing": ["a first-person video of boxing", "a boxing sparring match seen from the boxer's view"],
    "cooking": ["a first-person video of cooking in a kitchen", "hands preparing food on a kitchen counter",
                "a view of a kitchen sink, stove and cupboards"],
    "horse": ["a first-person video of horse riding", "a view between a horse's ears while riding"],
    "jumping": ["a first-person video of skydiving", "a POV video of jumping from a cliff or an aeroplane",
                "a person in freefall in the sky"],
    "longboarding": ["a first-person video of longboarding downhill on a road", "a POV video riding a longboard"],
    "motor": ["a first-person video of riding a motorcycle", "a POV video from a motorbike on a road or track"],
    "parkour": ["a first-person video of parkour", "a POV video of running and jumping over walls and rooftops"],
    "plane": ["a first-person video from the cockpit of a small plane", "a POV video of flying an aeroplane"],
    "rockclimbing": ["a first-person video of rock climbing", "hands gripping a rock wall while climbing"],
    "scuba": ["a first-person video of scuba diving underwater", "a POV video underwater with fish and coral"],
    "skate": ["a first-person video of skateboarding", "a POV video of a skateboard in a skate park"],
    "ski": ["a first-person video of skiing down a snowy slope", "a POV video of skis on snow"],
    "surfing": ["a first-person video of surfing a wave", "a POV video from a surfboard in the ocean"],
}


def label_prompts(label: str) -> list[str]:
    return LABEL_PROMPTS.get(label.lower(), [f"a first-person video of {human(label)}"])


# "Something is happening" vs "idle footage" contrasts used for the interaction cue.
KITCHEN_ACTIVE = [
    "hands cutting, pouring or stirring food",
    "hands holding and using a kitchen utensil",
    "a person opening a fridge or a cupboard",
    "hands washing dishes at a sink",
]
KITCHEN_IDLE = [
    "an empty kitchen counter with nothing happening",
    "a blurry or dark image",
    "a view of the ceiling or the floor",
    "a blank wall",
]
SPORT_ACTIVE = [
    "an exciting moment of a sport in action",
    "fast movement with a dramatic jump or turn",
    "a close-up of intense action",
]
SPORT_IDLE = [
    "a calm static view",
    "a blurry or dark image",
    "a view of the sky or the ground with nothing happening",
]


def interaction_prompts(domain: str) -> tuple[list[str], list[str]]:
    return (KITCHEN_ACTIVE, KITCHEN_IDLE) if domain == "kitchen" else (SPORT_ACTIVE, SPORT_IDLE)


PAIR_TEMPLATE = "a first-person photo of hands doing: {verb} {noun}"


def pair_prompt(verb: str, noun: str) -> str:
    return PAIR_TEMPLATE.format(verb=human(verb), noun=human(noun))
