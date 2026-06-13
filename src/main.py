"""
Main entry point for the MKWS project:
"Effect of Hydrogen Enrichment on Methane-Air Combustion:
Laminar Flame Speed, Ignition Delay and NOx Emissions"

Runs all three studies and saves figures (figures/) and raw data (data/).

Usage:
    python main.py            # run everything
    python main.py eq         # equilibrium study only
    python main.py flame      # flame speed study only (slow, ~10-15 min)
    python main.py ign        # ignition delay study only
"""

import sys

import equilibrium_study
import flame_speed_study
import ignition_delay_study


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "eq"):
        print("=" * 60)
        print("EQUILIBRIUM STUDY")
        print("=" * 60)
        equilibrium_study.run()

    if which in ("all", "flame"):
        print("=" * 60)
        print("LAMINAR FLAME SPEED STUDY")
        print("=" * 60)
        flame_speed_study.run()

    if which in ("all", "ign"):
        print("=" * 60)
        print("IGNITION DELAY STUDY")
        print("=" * 60)
        ignition_delay_study.run()


if __name__ == "__main__":
    main()
