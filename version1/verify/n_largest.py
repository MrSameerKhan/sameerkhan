


def n_largest():
    
    list_1 = [23,89,3645,952,10,20,30,50]
    n = 8
    final_list = []
    for i in range(1, n+1):
        
        larg = 0
        
        for j in range(len(list_1)):
            
            if list_1[j] > larg:
                larg = list_1[j]
                
        list_1.remove(larg)
        final_list.append(larg)
    print(final_list)
if __name__ == "__main__":
    n_largest()