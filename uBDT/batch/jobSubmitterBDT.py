from Condor.Production.jobSubmitter import *

class jobSubmitterBDT(jobSubmitter):
    def addExtraOptions(self,parser):
        super(jobSubmitterBDT,self).addExtraOptions(parser)
        
        parser.add_option("-C", "--configs", dest="configs", default="", help="input configuration(s), comma-separated (default = %default)")
        parser.add_option("-o", "--output", dest="output", default="", help="path to output directory in which root files will be stored (required) (default = %default)")

    def checkExtraOptions(self,options,parser):
        super(jobSubmitterBDT,self).checkExtraOptions(options,parser)

        if len(options.configs)==0:
            parser.error("Required option: --configs [C1,C2,...]")
        if len(options.output)==0:
            parser.error("Required option: --output [directory]")

    def generateExtra(self,job):
        super(jobSubmitterBDT,self).generateExtra(job)
        job.patterns.update([
            ("JOBNAME",job.name+"_$(Process)_$(Cluster)"),
            ("EXTRAINPUTS","input/args_"+job.name+"_$(Process).txt"),
            ("EXTRAARGS","-j "+job.name+" -p $(Process)"+" -i "+self.configs+" -o "+self.output),
        ])

    def generateSubmission(self):
        # create protojob
        job = protoJob()
        job.name = "trainBDT_"+self.configs
        self.generatePerJob(job)
        configs = self.configs.split(',')
        for iJob in range(len(configs)):
            job.njobs += 1
            job.nums.append(iJob)
        # append queue comment
        job.queue = "-queue "+str(job.njobs)
        # store protojob
        self.protoJobs.append(job)

    def finishedToJobName(self,val):
        return val.split("/")[-1].replace("training_","").replace(".tar.gz","")
