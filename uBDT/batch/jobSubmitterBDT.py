from Condor.Production.jobSubmitter import *

class jobSubmitterBDT(jobSubmitter):
    def addExtraOptions(self,parser):
        super(jobSubmitterBDT,self).addExtraOptions(parser)
        
        parser.add_option("-C", "--configs", dest="configs", type="string", action="callback", callback=list_callback, default="", help="input configuration(s), comma-separated (default = %default)")
        parser.add_option("-G", "--grid", dest="grid", type="int", default=-1, help="input grid version (default = %default)")
        parser.add_option("-o", "--output", dest="output", default="", help="path to output directory in which root files will be stored (required) (default = %default)")
        parser.add_option("-d", "--discard", dest="discard", default=False, action="store_true", help="make plots and discard large pkl files (default = %default)")

    def checkExtraOptions(self,options,parser):
        super(jobSubmitterBDT,self).checkExtraOptions(options,parser)

        if len(options.configs)==0:
            parser.error("Required option: --configs [C1,C2,...]")
        if len(options.configs)!=1 and options.grid>0:
            parser.error("Exactly one config required for grid mode")
        if len(options.output)==0:
            parser.error("Required option: --output [directory]")

    def generateExtra(self,job):
        super(jobSubmitterBDT,self).generateExtra(job)
        job.patterns.update([
            ("JOBNAME",job.name+"_$(Process)_$(Cluster)"),
            ("EXTRAINPUTS",""),
            ("EXTRAARGS","-j "+job.name+" -p $(Process)"+" -i "+self.configlist+(" -g" if self.grid>0 else "")+(" -d" if self.discard else "")+" -o "+self.output),
        ])

    def generateSubmission(self):
        # create protojob
        job = protoJob()
        if self.grid>0: # grid case
            # import grid info
            import sys
            sys.path.insert(0,'..')
            from mods import config_path
            config_path('..')
            from makeGrid import getGridName, makeGridPoints

            self.configlist = getGridName(self.configs[0],self.grid)
            job.name = "trainBDT_"+self.configlist
            self.generatePerJob(job)
            for iJob in range(len(makeGridPoints(self.grid))):
                job.njobs += 1
                job.nums.append(iJob)
        else:
            self.configlist = ','.join(self.configs)
            job.name = "trainBDT_"+self.configlist
            self.generatePerJob(job)
            for iJob in range(len(self.configs)):
                job.njobs += 1
                job.nums.append(iJob)
        # append queue comment
        job.queue = "-queue "+str(job.njobs)
        # store protojob
        self.protoJobs.append(job)

    def finishedToJobName(self,val):
        return val.split("/")[-1].replace("training_","").replace(".tar.gz","")
