"""Zarr-v3 I/O helpers shared by the stack encode/decode scripts.

Thin wrappers around tensorstore (imported lazily — only zarr paths need the
py38zarr env) plus the OME-NGFF 0.5 group metadata writer. Not part of the
core inference API.
"""

import json
import os


def open_zarr(store, level):
    """Open one pyramid level of a zarr-v3 store read-only -> TensorStore."""
    import tensorstore as ts
    return ts.open({'driver': 'zarr3',
                    'kvstore': {'driver': 'file',
                                'path': os.path.join(store, str(level))}},
                   read=True).result()


def create_zarr(path, shape, chunks, dtype='uint16', overwrite=False):
    """Create (or open, validating metadata) a writable zarr-v3 array.

    blosc-zstd compressed, matching the encode-side source store. The default
    opens an existing array (metadata must match), so successive chunk runs
    fill one store incrementally; overwrite=True wipes it first.
    """
    import tensorstore as ts
    return ts.open({'driver': 'zarr3',
                    'kvstore': {'driver': 'file', 'path': path},
                    'create': True,
                    'open': not overwrite,
                    'delete_existing': overwrite,
                    'metadata': {
                        'shape': list(shape),
                        'data_type': dtype,
                        'chunk_grid': {'name': 'regular',
                                       'configuration': {'chunk_shape': list(chunks)}},
                        'codecs': [{'name': 'bytes',
                                    'configuration': {'endian': 'little'}},
                                   {'name': 'blosc',
                                    'configuration': {'cname': 'zstd', 'clevel': 3,
                                                      'shuffle': 'shuffle'}}],
                    }}).result()


def write_ome_group(store, name, axes=('z', 'x', 'y')):
    """Write the group zarr.json that makes {store}/0 a single-scale OME-NGFF
    0.5 multiscale — same style as the encode-side source store."""
    meta = {'attributes': {'ome': {'version': '0.5', 'multiscales': [{
                'name': name,
                'axes': [{'name': a, 'type': 'space'} for a in axes],
                'datasets': [{'path': '0', 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [1.0] * len(axes)}]}]}]}},
            'zarr_format': 3, 'node_type': 'group'}
    with open(os.path.join(store, 'zarr.json'), 'w') as f:
        json.dump(meta, f, indent=2)
