import getopt, sys
from heatmap import *

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:m:', ['inpath=','outpath=','modelpath='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    imagepath,outpath,modelpath = '','',''

    for o, a in opts:
        if o in ('-i', '--inpath') and type(a)==str:
            imagepath = a
        elif o in ('-o', '--outpath') and type(a)==str:
            outpath = a 
        elif o in ('-m', '--modelpath') and type(a)==str:
            modelpath = a
        else:
            assert False, 'unhandled option'
    
    Heatmap().read_model(modelpath)
    Heatmap().image_array(imagepath)
    Heatmap().create_heatmap()
    Heatmap().display_heatmap(imagepath,outpath)
    
if __name__ == "__main__":
    main()
