import os
import time

for filename in os.listdir('.'):
    if os.path.splitext(filename)[-1] == '.csv':
        input_file = open(filename, 'r')
        output_file = open(os.path.join('./temp/', filename), 'w')
        
        #print 'Processing %s' % filename
        
        header = input_file.readline()
        time_type = header.split(',')[0]
        
        if time_type == '"Time"':
            time_format = '"%Y-%m-%d %H:%M:%S"'
        elif time_type == '"Date"':
            time_format = '"%Y-%m-%d"'
        elif time_type == '"Month"':
            time_format = '"%Y-%m"'
            
        for line in input_file:
            try:
                if line.strip() == '':
                    break
                atime = time.strptime(line.split(',')[0], time_format)
                time_value = atime.tm_year + (atime.tm_yday / 365.0) + (atime.tm_hour / (365.0 * 24)) + (atime.tm_min / (365.0 * 24 * 60))  + (atime.tm_sec / (365.0 * 24 * 60 * 60))
                value = line.split(',')[-1]
                output_file.write('%f,%s' % (time_value, value))
            except:
                print 'Warning : could not parse %s' % line.rstrip()
            
        input_file.close()
        output_file.close()
            
            
                
