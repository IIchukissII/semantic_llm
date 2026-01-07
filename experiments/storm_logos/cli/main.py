"""Main CLI entry point."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Storm-Logos: Adaptive Semantic System',
        prog='storm-logos',
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--genre', '-g', default='balanced',
                            choices=['dramatic', 'ironic', 'balanced'],
                            help='Genre to generate')
    gen_parser.add_argument('--sentences', '-n', type=int, default=3,
                            help='Number of sentences')
    gen_parser.add_argument('--seed', '-s', help='Seed text')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text')
    analyze_parser.add_argument('text', nargs='?', help='Text to analyze')
    analyze_parser.add_argument('--file', '-f', help='File to analyze')

    # Therapy command
    therapy_parser = subparsers.add_parser('therapy', help='Therapy session')
    therapy_parser.add_argument('--interactive', '-i', action='store_true',
                                help='Interactive mode')

    # Info command
    subparsers.add_parser('info', help='Show system info')

    args = parser.parse_args()

    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'therapy':
        cmd_therapy(args)
    elif args.command == 'info':
        cmd_info()
    else:
        parser.print_help()


def cmd_generate(args):
    """Generate command."""
    from ..applications.generator import Generator

    print(f"Generating {args.sentences} sentences in {args.genre} style...")
    print()

    generator = Generator()
    text = generator.generate(
        genre=args.genre,
        n_sentences=args.sentences,
        seed=args.seed,
    )

    print(text)


def cmd_analyze(args):
    """Analyze command."""
    from ..applications.analyzer import Analyzer

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Enter text to analyze (Ctrl+D to finish):")
        text = sys.stdin.read()

    analyzer = Analyzer()
    result = analyzer.analyze_text(text)

    print("\n=== Analysis ===")
    print(f"Position: A={result['position']['A']:+.2f}, "
          f"S={result['position']['S']:+.2f}, "
          f"τ={result['position']['tau']:.2f}")
    print(f"Coherence: {result['coherence']:.2f}")
    print(f"Irony: {result['irony']:.2f}")
    print(f"Tension: {result['tension']:.2f}")

    if result['defenses']:
        print(f"Defenses: {', '.join(result['defenses'])}")

    if result['dialectic']:
        dial = result['dialectic']
        print(f"\nThesis: {dial.get('thesis', {}).get('description', '')}")
        print(f"Antithesis: {dial.get('antithesis', {}).get('description', '')}")


def cmd_therapy(args):
    """Therapy command."""
    from ..applications.therapist import Therapist

    therapist = Therapist()

    if args.interactive:
        print("=== Storm-Logos Therapy Session ===")
        print("Type 'quit' to exit\n")

        while True:
            try:
                patient_input = input("Patient: ").strip()
                if patient_input.lower() in ('quit', 'exit', 'q'):
                    break
                if not patient_input:
                    continue

                response = therapist.respond(patient_input)
                print(f"\nTherapist: {response}\n")

            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
    else:
        print("Enter patient text (Ctrl+D to finish):")
        patient_text = sys.stdin.read().strip()
        if patient_text:
            response = therapist.respond(patient_text)
            print(f"\nTherapist: {response}")


def cmd_info():
    """Info command."""
    from .. import __version__
    from ..data.postgres import get_data

    print(f"Storm-Logos v{__version__}")
    print()
    print("Layers:")
    print("  1. DATA LAYER - PostgreSQL + Neo4j")
    print("  2. SEMANTIC LAYER - Storm, Dialectic, Chain")
    print("  3. METRICS ENGINE - Extractors + Analyzers")
    print("  4. FEEDBACK ENGINE - Error computation")
    print("  5. GENERATION ENGINE - Bond pipeline")
    print("  6. ADAPTIVE CONTROLLER - PI control")
    print("  7. ORCHESTRATION LAYER - Main loop")
    print("  8. APPLICATION LAYER - Therapist, Generator, etc.")
    print()

    try:
        data = get_data()
        stats = data.stats()
        print(f"Data: {stats['n_coordinates']:,} coordinates loaded")
    except Exception as e:
        print(f"Data: Error loading ({e})")


if __name__ == '__main__':
    main()
