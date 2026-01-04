# Freestyle > 2025-10-06 12:27pm
https://universe.roboflow.com/2bfound-ai-project/freestyle-kkgfa

Provided by a Roboflow user
License: CC BY 4.0


    1.The Human-to-Machine Formula

    The AI doesn't see "Pixels" (which change with screen size); it sees "Location Scores" (Percentages).

        Percentage×Image Dimension=Actual Pixel

        Example: If your photo is 848 pixels wide and the label says 0.50, the joint is exactly at pixel 424.


    2. The Data Template (The "Address" System)
        Every row of data is a single sentence. Let's split it into three sections:
        A Class
        B Box The Zone: A square around the person.
        C Joint The Spot: An exact body part triplet.

    3.Data explanation:

        each label contain one class and bounding box of the swimmer and 12 key points

        keypoints:(x,y,v) x,y are cordinate the data represent by precent of the picture so if x=500 we dvide by 848 and y in 340 height
        and v is visability the v is seprate by 0 not found,1 messy,2 clear

    5. Practical Example (Decoding your line)

    0 0.479 0.627 0.633 0.303 ... 0.608 0.549 2

        The Zone: "I found a swimmer (0) near the center (0.47, 0.62) covering about 60% of the horizontal view."

        The Head: "The nose is at 60% width and 54% height."

        The Visibility: "The score is 2, so the head is clearly above water in this frame."



    1. The Human-to-Machine Formula

    The AI uses Percentages (Location Scores) instead of pixels so it can work on any screen size. To find where a dot is on your specific screen, use this math:

        Percentage (0.xxx)×Image Dimension (Pixels)=Actual Pixel

        Example (X-axis): 0.479×848 pixels=406 (The point is at pixel 406 horizontally).

        Example (Y-axis): 0.627×480 pixels=300 (The point is at pixel 300 vertically).

    2. The Data Row Template

    Every line of data is a "sentence" with three parts. Here is how to read it:
    
    A: Class	0	Identity: "I found a swimmer."

    B: Box	0.479 0.627 0.633 0.303	The Zone: A box containing the swimmer (Center X, Center Y, Width, Height).

    C: Joint	0.608 0.549 2	The Spot: One body part (X, Y, Visibility).


    3. The "Connect-the-Dots" Map (Topology)

    The AI reads the body parts in a strict order. To analyze a stroke, you need to know which triplet is which:

        Index 0: Nose / Head

        Index 1 & 2: Shoulders (Left, Right)

        Index 3 & 4: Elbows (Left, Right)

        Index 5 & 6: Wrists (Left, Right)

        Index 7 - 12: Lower Body (Hips, Knees, Ankles)

    4. The Visibility "Sense" (V)

    This is the most critical part for your Swim Advisor logic. It tells you if you can trust the data for that specific frame:

        V = 2 (Clear): The joint is above water. Trust this for precise angle calculation.

        V = 1 (Messy): The joint is underwater or behind a splash. The AI is "guessing" based on where the shoulder is. Use this to understand general movement, but expect slight noise.

        V = 0 (Gone): The joint is not in the photo. Ignore this point for analysis.


    5. Summary of Your Line

    0 0.479 0.627 0.633 0.303 ... 0.608 0.549 2

        The Zone: "I found a swimmer centered near the middle of the pool, taking up most of the horizontal frame."

        The Head (Keypoint 0): "The nose is at 60.8% width and 54.9% height."

        The Status: "The score is 2, meaning the head is clearly visible (likely taking a breath or just above the surface)."