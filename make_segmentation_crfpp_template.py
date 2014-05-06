#!/usr/bin/env python3

def main():
    num_feature_types = 13

    with open('segmentation_crfpp_template.txt', 'w') as outfile:
        for i in range(num_feature_types):
            for j in [-2, -1, 0, 1, 2]:
                print('U{:03d}{}:%x[{},{}]'.format(i, j + 2, j, i), file=outfile)
            print(file=outfile)



if __name__ == '__main__':
    main()
