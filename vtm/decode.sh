#!/bin/sh

#./vtm/bin/VTM10Dec -c ./vtm/cfg/encoder_lowdelay_vtm.cfg -c ./vtm/cfg/per-sequence/43.cfg -c ./vtm/cfg/formatRGB.cfg -q $2 -i $1_org.rgb -wdt $3 -hgt $4 -o $1_rec.rgb -b $1.bin >>$1.log 

./vtm/bin/VTM10Dec -b $1.bin -o $1_dec.rgb --OutputColourSpaceConvert=GBRtoRGB > $1_dec.log



#./vtm/bin/EncoderCLIC -c ./vtm/cfg/encoder_lowdelay_clic.cfg -c ./vtm/cfg/per-sequence/50.cfg -q $2 -i $1_org.yuv -wdt $3 -hgt $4 -wdtc $5 -hgtc $6 -o $1_rec.yuv -b $1.bin >>$1.log 
#DecoderApp.exe -b ./bin/1.bin -o ./Dec_rgb/1.rgb --OutputColourSpaceConvert=GBRtoRGB > ./Dec_txt/1.txt
