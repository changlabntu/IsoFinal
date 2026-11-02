"""Self-containment check: every local import in IsoFinal resolves to a file
inside IsoFinal, and no source string points back at the parent repo or the
experiment trees. Run from anywhere:  python tools/check_closure.py"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = {d for d in os.listdir(ROOT)
         if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith('.')
         and any(f.endswith('.py') for _, _, fs in os.walk(os.path.join(ROOT, d)) for f in fs)}
FORBIDDEN = ('IsoScopeXX', 'latent_blending', 'inference_latent', 'latent_tsne',
             'ldm.modules.losses', 'edclean', 'ed023eMSdet', 'dsmc')


def module_exists(dotted):
    parts = dotted.split('.')
    base = os.path.join(ROOT, *parts)
    return os.path.isfile(base + '.py') or os.path.isdir(base)


def main():
    bad, external = [], set()
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__')]
        for f in fs:
            if not f.endswith('.py'):
                continue
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, ROOT)
            src = open(path).read()
            tree = ast.parse(src, rel)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for n in names:
                    top = n.split('.')[0]
                    if top in LOCAL:
                        if not module_exists(n):
                            bad.append('%s: import %s -> not in IsoFinal' % (rel, n))
                    else:
                        external.add(top)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for word in FORBIDDEN:
                        if word in node.value and rel != 'tools/check_closure.py':
                            bad.append('%s: string mentions %r' % (rel, word))
    print('local packages: %s' % ', '.join(sorted(LOCAL)))
    print('external imports: %s' % ', '.join(sorted(external)))
    if bad:
        print('\n'.join(bad))
        print('FAIL: %d problem(s)' % len(bad))
        sys.exit(1)
    print('OK: closure is self-contained')


if __name__ == '__main__':
    main()
