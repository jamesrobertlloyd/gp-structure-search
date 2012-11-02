'''
Created on Sep, 2012

Run shell scripts in a folder, but do not overload processor

@author: James Lloyd
'''

#### Written for speed accuracy tests
#### Don't need everything - might just be useful for inspiration

import subprocess, os, psutil, sys, time

def main ():
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = './'

    file_list = os.listdir('./')
    file_list.sort()

    for file_name in file_list:
        #print "Considering " + file_name
        #print file_name[-3:]
        if file_name[-3:] == '.sh':
            #print file_name + " is a shell script"
            print "CPU percent is %2.1f" % psutil.cpu_percent()
            # We have a shell script
            while psutil.cpu_percent() > 40:
                print "Sleeping"
                time.sleep(60)
            # CPU not overused - run the script in a separate process
            #print "I will tell subprocess to run "
            my_call = ["/misc/apps/matlab/matlabR2011b/bin/matlab"] + open(file_name, 'r').read().split(' ')[1:]
            #print my_call
            my_file = open(file_name.split('.')[0] + '.out', 'w')
            subprocess.Popen(my_call, stdout = my_file);
            #subprocess.Popen(my_call);
            #print os.system("matlab -nosplash -nojvm -nodisplay -r jura_optim_m004")
            #subprocess.call(["/usr/local/apps/matlab/matlabR2011b/bin/matlab"])
            #subprocess.call(["/usr/local/apps/matlab/matlabR2011b/bin/matlab", "-nosplash", "-nojvm", "-nodisplay", "-r", "jura_optim_m004"])
            #subprocess.call(["./" + file_name])
            #subprocess.call(["ls"])
            print "Running " + file_name
            time.sleep(5)
    

if __name__ == '__main__':
    main()
    print 'Goodbye, World!'