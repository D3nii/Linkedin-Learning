import face_recognition

# Load the known images
image1 = face_recognition.load_image_file('person1.jpg')
image2 = face_recognition.load_image_file('person2.jpg')
image3 = face_recognition.load_image_file('person3.jpg')

# Get the face encoding of each person. This can fail if no one is found in the photo.
person1 = face_recognition.face_encodings(image1[0])
person2 = face_recognition.face_encodings(image2[0])
person3 = face_recognition.face_encodings(image3[0])

# Create a list of all known face encodings
known_face_encodings = [
    person1,
    person2,
    person3
]

# Load the image we want to check
unknown = face_recognition.load_image_file('unknown_1.jpg')

# Get face encodings for any people in the picture
unknown_encodings = face_recognition.face_encodings(unknown)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_encodings)
    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"

    print(f"Found {name} in the photo!")
