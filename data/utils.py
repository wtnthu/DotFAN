from pathlib import Path
import os
def read_txt(name):
    lines = Path(name).read_text().strip().split('\n')
    return lines
def write_txt(name,list,changeline=True):
    f = open(name,'w')
    for item in list:
        if changeline:
             f.write("%s\n" % item)
        else:
            f.write("%s" % item)
    f.close()
def find_dir(path):
    out = ''
    list_ = path.split('/')
    for i in range(len(list_)-1):
        out += list_[i]+'/'
    return out[:-1]
def read_folder(Folder_root,WithFolder_root=False):
    file_list = []
    for dirname, dirnames, filenames in os.walk(Folder_root):
        for filename in filenames:
            if WithFolder_root:
                file_list.append(os.path.join(dirname, filename))
            else:
                file_list.append(os.path.join(dirname, filename).split(Folder_root+"/")[1])
    return  file_list
if __name__=="__main__":
    #file_list = read_folder('/home/kangyu/Dataset/large_Face_dataset/faces_emore/imgs_casia')
    tests = read_txt('test_all_celebA.txt')[1:]
    final = []
    rem_dict = {}
    for test in tests:
        if not test.split('/')[0] in rem_dict.keys():
            rem_dict[test.split('/')[0]] = []
        if len(rem_dict[test.split('/')[0]])<13:
            rem_dict[test.split('/')[0]].append(test)
            final.append(test)

    write_txt('test_celeb_a13.txt',final)