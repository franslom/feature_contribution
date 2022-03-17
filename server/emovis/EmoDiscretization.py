"""
Based on the paper titled "Emotion Classification in Arousal Valence Model using MAHNOB-HCI Database"
Link: https://pdfs.semanticscholar.org/3750/b635d455fee489305b24ead4b7e9233b7209.pdf
"""

quadrants = {
    4: [
        {'name': 'PV-HA', 'desc': 'Positive Valence - High Arousal', 'center': [7, 7]},
        {'name': 'NV-HA', 'desc': 'Negative Valence - High Arousal', 'center': [3, 7]},
        {'name': 'NV-LA', 'desc': 'Negative Valence - Low Arousal', 'center': [3, 3]},
        {'name': 'PV-LA', 'desc': 'Positive Valence - Low Arousal', 'center': [7, 3]}
    ],
    9: [
        {'name': 'PL-EX', 'desc': 'Pleasant - Excited', 'center': [8, 8]},
        {'name': 'NE-EX', 'desc': 'Neutral - Excited (Surprise)', 'center': [5, 8]},
        {'name': 'UN-EX', 'desc': 'Unpleasant - Excited (Anger, Anxiety, Fear)', 'center': [2, 8]},
        {'name': 'PL-ME', 'desc': 'Pleasant - Medium (Happiness, Amusement)', 'center': [8, 5]},
        {'name': 'NE-ME', 'desc': 'Neutral - Medium', 'center': [5, 5]},
        {'name': 'UN-ME', 'desc': 'Unpleasant - Medium', 'center': [2, 5]},
        {'name': 'PL-CL', 'desc': 'Pleasant - Calm', 'center': [8, 2]},
        {'name': 'NE-CL', 'desc': 'Neutral', 'center': [5, 2]},
        {'name': 'UN-CL', 'desc': 'Unpleasant - Calm (Sadness, Disgust)', 'center': [2, 2]}
    ]
}


def discretize_by_quadrant(labels, n_classes=4):
    if n_classes == 4:
        for i in range(len(labels)):
            if labels[i]['valence'] >= 4.5:     # Positive
                if labels[i]['arousal'] >= 4.5: # High
                    labels[i]['emotion'] = 0
                else:
                    labels[i]['emotion'] = 3
            else:   # Negative
                if labels[i]['arousal'] >= 4.5: # High
                    labels[i]['emotion'] = 1
                else:
                    labels[i]['emotion'] = 2
    elif n_classes == 9:
        for i in range(0, len(labels)):
            if 1 <= labels[i]['arousal'] <= 3:  # Calm
                if 1 <= labels[i]['valence'] <= 3:  # Unpleasant
                    labels[i]['emotion'] = 8
                elif 4 <= labels[i]['valence'] <= 6:  # Neutral
                    labels[i]['emotion'] = 7
                elif 7 <= labels[i]['valence'] <= 9:  # Pleasant
                    labels[i]['emotion'] = 6
            elif 4 <= labels[i]['arousal'] <= 6:  # Medium
                if 1 <= labels[i]['valence'] <= 3:  # Unpleasant
                    labels[i]['emotion'] = 5
                elif 4 <= labels[i]['valence'] <= 6:  # Neutral
                    labels[i]['emotion'] = 4
                elif 7 <= labels[i]['valence'] <= 9:  # Pleasant
                    labels[i]['emotion'] = 3
            elif 7 <= labels[i]['arousal'] <= 9:  # Excited
                if 1 <= labels[i]['valence'] <= 3:  # Unpleasant
                    labels[i]['emotion'] = 2
                elif 4 <= labels[i]['valence'] <= 6:  # Neutral
                    labels[i]['emotion'] = 1
                elif 7 <= labels[i]['valence'] <= 9:  # Pleasant
                    labels[i]['emotion'] = 0
    return labels, quadrants[n_classes]


levels = {
    4: {'arousal': ['Low', 'High'], 'valence': ['Negative', 'Positive'], 'center': [3, 7]},
    9: {'arousal': ['Calm', 'Medium', 'Excited'], 'valence': ['Unpleasant', 'Neutral', 'Pleasant'], 'center': [2, 5, 8]},
}


def discretize_by_level(labels, n_classes):
    if n_classes == 4:
        for i in range(len(labels)):
            if labels[i]['valence'] <= 4.5: # Positive
                labels[i]['val_lvl'] = 0
            else:
                labels[i]['val_lvl'] = 1

            if labels[i]['arousal'] <= 4.5: # High
                labels[i]['aro_lvl'] = 0
            else:
                labels[i]['aro_lvl'] = 1
    elif n_classes == 9:
        for i in range(0, len(labels)):
            if 1 <= labels[i]['arousal'] <= 3:      # Calm
                labels[i]['aro_lvl'] = 0
            elif 4 <= labels[i]['arousal'] <= 6:    # Medium
                labels[i]['aro_lvl'] = 1
            elif 7 <= labels[i]['arousal'] <= 9:    # Excited
                labels[i]['aro_lvl'] = 2
                
            if 1 <= labels[i]['valence'] <= 3:  # Unpleasant
                labels[i]['val_lvl'] = 0
            elif 4 <= labels[i]['valence'] <= 6:  # Neutral
                labels[i]['val_lvl'] = 1
            elif 7 <= labels[i]['valence'] <= 9:  # Pleasant
                labels[i]['val_lvl'] = 2
    return labels, levels[n_classes]


def get_centroid_emotion(emotion, n_classes):
    center = quadrants[n_classes][emotion]['center']
    return {'valence': center[0], 'arousal': center[1], 'emotion': emotion}


def get_centroid_level(level, n_classes):
    center = levels[n_classes]['center'][level]
    return center
