set -e

packages="export.messages export.subscriber export.export visualization models utils"
dotdir="diagrams"
if [ ! -d $dotdir ]; then
    mkdir $dotdir
else
    rm -rf $dotdir/*
fi

for subpackage in $packages
do
    (
    fname="ikkuna.$subpackage"
    echo "Writing dot file to $dotdir/$fname.dot"
    pyreverse -o dot --module-names=y --only-classnames -p $fname ikkuna/$subpackage

    # classes_… is always generated
    mv classes_$fname.dot $dotdir
    # somehow, passing -Grankdir=LR doesn't to jack shit, so use sed #yolo
    sed -i "s/rankdir=../rankdir=LR/g" $dotdir/classes_$fname.dot
    echo "Generating class diagram for $fname"
    dot -Tpdf -Nfontname=DejaVuSansMono -Efontname=DejaVuSansMono $dotdir/classes_$fname.dot > "$dotdir/classes_ikkuna.$subpackage.pdf"

    # packages_… file is not generated for every file
    if [ -f packages_$fname.dot ]; then
        mv packages_$fname.dot $dotdir
        sed -i "s/rankdir=../rankdir=LR/g" $dotdir/packages_$fname.dot
        echo "Generating package diagram for $fname"
        dot -Tpdf -Nfontname=DejaVuSansMono -Efontname=DejaVuSansMono $dotdir/packages_$fname.dot > "$dotdir/packages_ikkuna.$subpackage.pdf"
    fi
    ) &
done
wait
