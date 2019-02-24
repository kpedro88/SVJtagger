from jobSubmitterBDT import jobSubmitterBDT

def submitJobs():  
    mySubmitter = jobSubmitterBDT()
    mySubmitter.run()
    
if __name__=="__main__":
    submitJobs()
