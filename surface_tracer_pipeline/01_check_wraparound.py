r"""
16-bit wraparound in the acquired images: how many voxels, where, and how much signal is at stake.

The raw stacks are saved as **signed** int16, so any voxel brighter than 32,767 wraps to a large
negative number (45,148 is stored as -20,388). This is present in the raw files, not introduced by
the pipeline - anX/fitc.tif is already int16 - so it happened at acquisition or export.

It matters out of proportion to its size:
  * the wrapped voxels are the BRIGHTEST ones, so they are exactly the tracer we care about;
  * being negative they fall below every threshold, so they are dropped from the tracer mask and
    from any intensity sum, biasing against the animals with most signal;
  * mask_generation._calculate_threshold() builds a 256-bin histogram and takes the mode of the
    first 50 bins. With a range stretched from -25,000 to +27,000 those bins are meaningless, and
    the threshold it returns can be negative (an36: -13,951), which makes the pipeline's brain mask
    threshold everything and leaves the convex hull to define the mask.

Downsampling used skimage.resize(anti_aliasing=True, order=1), which averages neighbouring voxels,
so a wrapped value has already been blended into its neighbourhood: the true intensity cannot be
recovered from the downsampled data. Repair requires re-downsampling from the raw stacks with the
wrap undone (add 65536 to negatives) - expensive, and only worth it if the affected voxels matter
for a particular measurement.

Run:  python check_wraparound.py
"""
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

HERE = os.path.dirname(os.path.abspath(__file__))

PARENT = config.DATA_DIR
QC_DIR = config.QC_DIR
CHANNELS = {'reference': config.REFERENCE_IMAGE,
            **{name: f'{stem}.tif' for name, stem in config.CHANNELS.items()}}
TREATMENT_ORDER = config.GROUP_ORDER
GROUP_COLORS = config.GROUP_COLORS
EXAMPLE_ANIMAL = config.EXAMPLE_ANIMAL


def scan():
    """Per animal x channel: how many voxels wrapped, and what the extremes are."""
    data_map = config.load_data_map()
    animals = config.animals()
    rows = []
    for animal in animals:
        for channel, filename in CHANNELS.items():
            path = os.path.join(PARENT, animal, 'downsampled', filename)
            if not os.path.exists(path):
                continue
            volume = tifffile.imread(path)
            negative = volume < 0
            n = int(negative.sum())
            rows.append({'animal': animal, 'treatment': data_map.loc[animal, 'treatment'], 'channel': channel,
                         'dtype': str(volume.dtype), 'min': int(volume.min()), 'max': int(volume.max()),
                         'n_wrapped': n, 'ppm': 1e6 * n / volume.size,
                         'lost_signal_estimate': float(np.abs(volume[negative].astype(np.float64) + 65536).sum()) if n else 0.0,
                         'total_positive_signal': float(volume[volume > 0].astype(np.float64).sum())})
            del volume, negative
        print(f'  {animal} done', flush=True)
    return pd.DataFrame(rows)


def figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={'width_ratios': [1.6, 1, 1]})

    # how many voxels wrapped, per animal and channel
    ax = axes[0]
    animals = sorted(df['animal'].unique(), key=config.sort_key)
    width = 0.27
    for k, channel in enumerate(CHANNELS):
        d = df[df['channel'] == channel].set_index('animal').reindex(animals)
        ax.bar(np.arange(len(animals)) + (k - 1) * width, d['n_wrapped'].fillna(0) + 0.5, width=width,
               label=channel, edgecolor='#333333', linewidth=0.4)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(animals)))
    ax.set_xticklabels(animals, rotation=90, fontsize=8)
    for tick, animal in zip(ax.get_xticklabels(), animals):
        tick.set_color(GROUP_COLORS[df.loc[df['animal'] == animal, 'treatment'].iloc[0]])
    ax.set_ylabel('wrapped voxels (log, 0.5 = none)', fontsize=10)
    ax.set_title('Voxels above 32,767 that wrapped negative', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # how close each animal runs to the 16-bit ceiling
    ax = axes[1]
    for treatment in TREATMENT_ORDER:
        d = df[(df['treatment'] == treatment) & (df['channel'] != 'reference')]
        ax.scatter(d['max'], d['n_wrapped'] + 0.5, s=45, color=GROUP_COLORS[treatment],
                   edgecolor='black', linewidth=0.6, label=config.short(treatment))
    ax.axvline(32767, color='#C1666B', linestyle='--', linewidth=1.2)
    ax.text(32767, ax.get_ylim()[1], ' int16 ceiling', color='#C1666B', fontsize=9, va='top')
    ax.set_yscale('log')
    ax.set_xlabel('brightest surviving value in the stack', fontsize=10)
    ax.set_ylabel('wrapped voxels', fontsize=10)
    ax.set_title('Every affected stack sits at the ceiling', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # where they are, in the worst animal
    ax = axes[2]
    first_channel = CHANNELS[list(config.CHANNELS)[0]]
    volume = tifffile.imread(os.path.join(PARENT, EXAMPLE_ANIMAL, 'downsampled', first_channel))
    negative = volume < 0
    plane = int(np.argmax(negative.sum(axis=(1, 2)))) if negative.any() else volume.shape[0] // 2
    image = volume[plane].astype(np.float32)
    ax.imshow(image, cmap='gray', vmin=0, vmax=np.percentile(image[image > 0], 99.5), interpolation='nearest')
    ys, xs = np.nonzero(negative[plane])
    ax.scatter(xs, ys, s=90, facecolors='none', edgecolors='#66CCFF', linewidths=1.2)
    ax.set_title(f'{EXAMPLE_ANIMAL}, plane {plane}: wrapped voxels circled', fontsize=11, loc='left')
    ax.set_axis_off()

    fig.suptitle('16-bit wraparound in the acquired stacks (signed int16, values above 32,767 stored negative)',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(QC_DIR, exist_ok=True)
    path = os.path.join(QC_DIR, 'int16_wraparound.png')
    plt.savefig(path, dpi=120)
    print(f'wrote {path}')


def main():
    df = scan()
    df.to_csv(os.path.join(QC_DIR, 'int16_wraparound.csv'), index=False)
    pd.set_option('display.width', 220)
    affected = df[df['n_wrapped'] > 0]
    print(f'\n{affected["animal"].nunique()} of {df["animal"].nunique()} animals affected, '
          f'{len(affected)} of {len(df)} stacks')
    print('\nworst 12 stacks:')
    print(affected.nlargest(12, 'n_wrapped')[['animal', 'treatment', 'channel', 'min', 'max', 'n_wrapped', 'ppm']]
          .to_string(index=False))
    print('\nby group (mean wrapped voxels per stack, tracer channels only):')
    print(df[df['channel'] != 'reference'].groupby(['treatment', 'channel'])['n_wrapped'].mean()
          .unstack().reindex(TREATMENT_ORDER).round(1).to_string())
    print('\nestimated signal lost to wrapping, as a fraction of each stack\'s total (tracer channels):')
    tracer = df[df['channel'] != 'reference'].copy()
    tracer['lost_fraction_%'] = 100 * tracer['lost_signal_estimate'] / tracer['total_positive_signal']
    print(tracer.nlargest(8, 'lost_fraction_%')[['animal', 'treatment', 'channel', 'n_wrapped', 'lost_fraction_%']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    figure(df)


if __name__ == '__main__':
    main()
