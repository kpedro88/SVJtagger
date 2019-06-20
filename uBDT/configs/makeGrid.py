from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

grids = OrderedDict([
    (1, OrderedDict([
            ("n_estimators", [100,500,1000,5000]),
            ("max_depth", [2,3,4]),
            ("learning_rate", [0.01,0.1,0.3,0.5,0.7,1.0]),
            ("power", [1.3,2.0]),
            ("fl_coefficient", [3,10,50]),
        ])
    ),
    (2, OrderedDict([
            ("n_estimators", [200,500,1000,2000]),
            ("max_depth", [2,3,4]),
            ("learning_rate", [0.01,0.1,0.3,0.5,0.7,1.0]),
            ("subsample", [0.3,0.6,0.9]),
            ("min_samples_leaf", [0.01,0.05,0.1]),
        ])
    ),
])

def getVars(gridversion=1):
    return grids[gridversion]

# recursive nested loop
def varyAll(paramlist,gridpoint,gridpoints,pos=0):
    param = paramlist[pos][0]
    vals = paramlist[pos][1]
    for v in vals:
        stmp = gridpoint[:]+[v]
        # check if last param
        if pos+1==len(paramlist):
            gridpoints.append(tuple(stmp))
        else:
            varyAll(paramlist,stmp,gridpoints,pos+1)

def getGridName(baseconfig,gridversion=1):
    gridname = baseconfig+"G{0:03d}".format(gridversion)
    return gridname

def getPointName(ipoint,basename,gridversion=0):
    if gridversion>0:
        gridname = getGridName(basename,gridversion)
    else:
        gridname = basename
    pointname = gridname+"P{0:04d}".format(ipoint)
    return pointname

def makeGridPoints(gridversion=1):
    gridvars = getVars(gridversion)
    gridpoints = []
    varyAll(list(gridvars.iteritems()),[],gridpoints)
    return gridpoints

def makeConfigs(baseconfig,gridversion=1):
    preamble = '\n'.join(["from mods import config_path","config_path()","from "+baseconfig+" import uconfig"])
    gridname = getGridName(baseconfig,gridversion)

    gridvars = getVars(gridversion)
    gridpoints = makeGridPoints(gridversion)

    for igp,gridpoint in enumerate(gridpoints):
        pointname = getPointName(igp,gridname)
        configlines = []
        for gridvar,gridval in zip(gridvars,gridpoint):
            configlines.append("uconfig.hyper."+gridvar+" = "+str(gridval))
        with open("grids/"+pointname+".py",'w') as outfile:
            outfile.write(preamble)
            outfile.write("\n\n")
            outfile.write('\n'.join(configlines))

if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-C","--config", dest="config", type=str, default="", help="base config to vary in grid (required)")
    parser.add_argument("-G","--gridversion", dest="gridversion", type=int, default=1, help="grid version")
    args = parser.parse_args()

    if len(args.config)==0:
        parser.error("Required argument: --config")

    makeConfigs(args.config,args.gridversion)
