from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('3p','2f'):
    raise SystemExit('usage: make_sine_53_xeon_x21_nearestpi.py {3p|2f}')
mode=sys.argv[1]

# Start from the current throughput champion: x20A = G4 one-pass + two gathers.
runpy.run_path('make_sine_53_xeon_x20_fused_batch.py', run_name='__main__', init_globals={'__name__':'__main__'})
# The parent script requires argv=2g; invoke it explicitly instead of relying on ours.
