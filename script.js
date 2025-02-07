// Sample data - replace with your actual content
const content = {
    note: [
        { title: "H-ll-W-rld", preview: "Do you like videogames?", image: "https://i.pinimg.com/736x/d4/09/b2/d409b254a9ad71f0225993123fea6840.jpg", link: "Note/helloworld.html" },
        // Add more notes...
    ],
    clip: [
        { title: "catto", image: "https://i.pinimg.com/564x/c4/93/20/c49320b072068b5da9b176d1a77adce7.jpg", link: "Clip/gallery.html" },
        // Add more clips...
    ],
    memory: [
        { title: "SoundWaves", preview: "Vibes..", image: "https://i.pinimg.com/564x/11/92/33/1192331815ef0cc6e9934d5c87aba5f3.jpg", link: "Memory/playlist.html" },
        // Add more memories...
    ],
    thought: [
        { title: "neuralnet", preview: "How a computer reads numbers:", image: "https://i.pinimg.com/564x/01/18/d1/0118d1ecf9211898da08a8225079c33b.jpg", link: "Thought/intelligence.html" },
        // Add more thoughts...
    ]
};

// Function to create content items
function createItem(item) {
    return `
        <div class="item">
            <img src="${item.image}" alt="${item.title}">
            <div class="item-content">
                <h3>${item.title}</h3>
                <p>${item.preview || ''}</p>
                <a href="${item.link}">Don't Click</a>
            </div>
        </div>
    `;
}

// Populate content
Object.keys(content).forEach(section => {
    const sectionEl = document.getElementById(section);
    content[section].forEach(item => {
        sectionEl.innerHTML += createItem(item);
    });
});
