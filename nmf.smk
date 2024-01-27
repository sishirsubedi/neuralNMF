import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim'
exp_dir = 'simdata/results/'



rule all:
    input:
        expand(exp_dir + sample+'.h5nnmf')

rule run_nnmf:
    input:
        script = 'experiment_nnmf.py',
        bulk_data = '/home/BCCRC.CA/ssubedi/projects/experiments/neuralNMF/simdata/data/sim.h5'
    output:
        res = exp_dir + sample+'.h5nnmf'
    shell:
        'python {input.script} {input.bulk_data}'