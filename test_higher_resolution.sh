#!/bin/bash
# Test colorization at higher resolution (512x512) for better face detail
# Usage: ./test_higher_resolution.sh

echo "Testing colorization at 512x512 resolution..."
echo "This will take longer but should improve small face detail."
echo ""

cd /Users/abhishekhallad/Documents/tmp-copy

# Test with 512x512
python3 test_fusion.py \
    --name test_fusion \
    --sample_p 1.0 \
    --model fusion \
    --fineSize 512 \
    --test_img_dir example \
    --results_img_dir results_512

echo ""
echo "✅ Higher resolution test complete!"
echo "Results saved in: results_512/"
echo ""
echo "Compare with 256x256 results in: results/"
echo "Higher resolution should show better detail for small faces."

