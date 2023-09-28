tt = zeros(20,1);
avt = zeros(3,1);
num = zeros(3,1);

tt10 = zeros(3,1);
tt101 = zeros(3,1);

for i = 1:3139
    for j=1:20
        if sqrt(mpx1(i,j)*mpx1(i,j)+mpy1(i,j)*mpy1(i,j))<2.5
            tt(j) = 1;
        end
    end
    if sum(tt)==20
        avt(1) = i;
        break
    else
        tt = zeros(20,1);
    end
end

tt = zeros(20,1);
for i = 1:21599
    for j=1:20
        if sqrt(mpx2(i,j)*mpx2(i,j)+mpy2(i,j)*mpy2(i,j))<2.5
            tt(j) = 1;
        end
    end
    if sum(tt)==20
        avt(2) = i;
        break
    else
        tt = zeros(20,1);
    end
end

tt = zeros(20,1);
for i = 1:21222
    for j=1:20
        if sqrt(mpx3(i,j)*mpx3(i,j)+mpy3(i,j)*mpy3(i,j))<2.5
            tt(j) = 1;
        end
    end
    if sum(tt)==20
        avt(3) = i;
        break
    else
        tt = zeros(20,1);
    end
end

mean(avt)

for j=1:20
   if sqrt(px1(13368,j)*px1(13368,j)+py1(13368,j)*py1(13368,j))<2.5
       num(1) = num(1)+1;
   end
   if sqrt(px2(13368,j)*px2(13368,j)+py2(13368,j)*py2(13368,j))<2.5
       num(2) = num(2)+1;
   end
   if sqrt(px3(13368,j)*px3(13368,j)+py3(13368,j)*py3(13368,j))<2.5
       num(3) = num(3)+1;
   end
end

mean(num)

hold on;
p = nsidedpoly(1000, 'Center', [0 0], 'Radius', 2.5);
hold on;
plot(p, 'FaceColor', 'g')
axis equal
hold on
scatter(mpx3(21222,1:20),mpy3(21222,1:20))


numt = zeros(3,1);
range = 3.5;
mem = 15;

for i = 1:3139
    for j=1:20
        if sqrt(mpx1(i,j)*mpx1(i,j)+mpy1(i,j)*mpy1(i,j))<range
            numt(1) = numt(1)+1;
        end
    end
    if numt(1)>mem
        tt101(1) = i;
        break
    else
        numt(1) = 0;
    end
end

for i = 1:21599
    for j=1:20
        if sqrt(mpx2(i,j)*mpx2(i,j)+mpy2(i,j)*mpy2(i,j))<range
            numt(2) = numt(2)+1;
        end
    end
    if numt(2)>mem
        tt101(2) = i;
        break
    else
        numt(2) = 0;
    end
end

for i = 1:21222
    for j=1:20
        if sqrt(mpx3(i,j)*mpx3(i,j)+mpy3(i,j)*mpy3(i,j))<range
            numt(3) = numt(3)+1;
        end
    end
    if numt(3)>mem
        tt101(3) = i;
        break
    else
        numt(3) = 0;
    end
end





numt = zeros(3,1);
for i = 1:17387
    for j=1:20
        if sqrt(px1(i,j)*px1(i,j)+py1(i,j)*py1(i,j))<range
            numt(1) = numt(1)+1;
        end
    end
    if numt(1)>mem
        tt10(1) = i;
        break
    else
        numt(1)= 0;
    end
end

for i = 1:13368
    for j=1:20
        if sqrt(px2(i,j)*px2(i,j)+py2(i,j)*py2(i,j))<range
            numt(2) = numt(2)+1;
        end
    end
    if numt(2)>mem
        tt10(2) = i;
        break
    else
        numt(2)= 0;
    end
end

for i = 1:13475
    for j=1:20
        if sqrt(px3(i,j)*px3(i,j)+py3(i,j)*py3(i,j))<range
            numt(3) = numt(3)+1;
        end
    end
    if numt(3)>mem
        tt10(3) = i;
        break
    else
        numt(3) = 0;
    end
end
