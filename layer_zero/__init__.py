"""
Layer Zero — Persona Minting Machine

Mints psychologically coherent Persona objects from user inputs
(text descriptions, structured fields, CSV segments, or direct specification)
for use with the persona-engine.

Usage:
    import layer_zero

    # From text
    personas = layer_zero.from_description("35-year-old nurse from Chicago", count=5)

    # From structured fields
    personas = layer_zero.mint(occupation="nurse", age=35, count=10)

    # From CSV segments
    personas = layer_zero.from_csv("segments.csv", count_per_segment=20)

    # Direct specification
    personas = layer_zero.mint(
        big_five={"openness": 0.8}, occupation="researcher", count=1
    )
"""

__version__ = "0.1.0"
