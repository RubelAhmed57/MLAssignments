target="2	2	2 1	1	1	2	2	1	1	1	2	2	1	2	2	1	1	2	2	2	2	2	1	1	2	2	1	1	2	1	2	2	1	2	1	2	2	1	1	1	2	2	2	2	2	2	1	1	2	2	2	1	1	2	1	1	2	2	2	2	2	1	2	2	2	1	2	2	1	2	1	1".split()

def cleaning(file):
    with open(file) as DataFile, open("Dataset.txt",'w') as D, open("Target.txt",'w') as Target:
        for line in DataFile:
            line=line.split()
            target=str(line[0].strip('"'))

            Target.write(target+'\n')
            D.write(str(line[1:62]).strip('[]')+'\n')


def transform(file):
    print len(target)
    with open(file,'w') as D:
        for item in target:
            D.write(item+'\n')


if __name__ == '__main__':
   transform("target.txt")