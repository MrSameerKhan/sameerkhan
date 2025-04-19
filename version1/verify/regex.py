
stringR = "aaaa{v?}a $1?{23ru{n?}kkkk"
z = stringR.replace('{', '', 3) 
y = z.replace('}','',3)
z = y.replace('?','',3)
    
print(z)