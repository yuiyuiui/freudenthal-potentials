"""Run every completed verification used by the critical-degree proof."""
from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parent
COMMANDS = [
    ['src/certify_uniform_normalform.py'],
    ['src/verify_repairs.py'],
    ['src/dimension_polynomial.py'],
    ['aim633_experiments/src/verify_saved.py', 'results/N1_d6',
     'aim633_experiments/results/N2_d6', 'aim633_experiments/results/N3_d6',
     'results/N4_d6'],
    ['src/transport_regression.py'],
]

def main() -> None:
    completed = []
    for args in COMMANDS:
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
        completed.append(args[0])
    (ROOT / 'results' / 'all_verifications.json').write_text(
        json.dumps({'passed': True, 'completed': completed,
                    'scope': 'all N at k=5; not all k>=5'}, indent=2)
    )

if __name__ == '__main__':
    main()
