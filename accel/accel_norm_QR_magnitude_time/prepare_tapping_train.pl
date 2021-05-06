#!/usr/bin/perl
#
## record tapping column
open REF, "/ssd/gyuanfan/2018/PDDB_tapping/Tapping_activity_training.tsv" or die;
while ($line=<REF>){
	chomp $line;
	$line=~s/"//g;
	@table=split "\t", $line;
	$rid=$table[2];
	$tapping_out=$table[7];
	if ($tapping_out eq ""){}else{
		$record{$rid}=$tapping_out;
	}
}
close REF;

open OLD, "train_gs.dat" or die;
open NEW, ">tapping_train.txt" or die;
while ($line=<OLD>){
	chomp $line;
	@table=split "\t", $line;
	if (-e "/ssd/gyuanfan/2018/PDDB_tapping/pre_processing/accel/$record{$table[0]}"){
		if ($table[0] eq ""){}else{
			print NEW "$table[1]\t/ssd/gyuanfan/2018/PDDB_tapping/pre_processing/accel/$record{$table[0]}\n";
	}
	}else{
		#print NEW "$table[1]\t\n";
	}

}

close OLD;
