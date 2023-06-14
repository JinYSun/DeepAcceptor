#!/work/jaydy/anaconda2/envs/my-rdkit-env/bin/python


from edud import CalculateSAscore


job_names = [_.rstrip() for _ in file("../data/edud_jobs.txt")]

ofs = open("../data/all_edud_smi_paths.txt", "w")
for job_name in job_names:
    task = CalculateSAscore(job_name)
    smi_fns = task.getAllSmiFiles()
    for fn in smi_fns:
        ofs.write(fn + "\n")
