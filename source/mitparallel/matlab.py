import subprocess


WRAPPER = """
exitcode = 0;
try
%(code)s
catch err
  exitcode = 1;
  rethrow(err);
end

exit(exitcode);
"""

def add_wrapper(code):
    lines = code.split('\n')
    lines = ['  %s' % line for line in lines]
    code = '\n'.join(lines)
    return WRAPPER % {'code': code}


TEST_CODE = """
error('fail!');
fprintf('hello world!');
"""

def run(code):
    code = add_wrapper(code)
    p = subprocess.Popen(['matlab', '-nodisplay', '-nosplash', '-nojvm'], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_value, stderr_value = p.communicate(code)
    p.wait()
    if p.returncode:
        raise RuntimeError('Error in Matlab code:\n\n%s' % stderr_value)


